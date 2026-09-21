/**
 * fusion-runtime browser client.
 *
 *   <script src="https://your-server/fusion-runtime.js"></script>
 *   <script>FusionRuntime.attach({ button: "#talk" });</script>
 *
 * With no `url`, it connects back to wherever this script was served from, so a
 * page only needs the two lines above.
 *
 * What it does
 * ------------
 * Microphone in: the browser's own echo canceller is on, so the agent's voice
 * doesn't come back as speech. Audio is resampled to the rate the server asks
 * for (16 kHz) inside an AudioWorklet — off the main thread, so a busy page
 * can't stutter the audio — and sent as raw 16-bit PCM.
 *
 * Reply out: PCM arrives in chunks and is scheduled back to back on a clock, a
 * little ahead of real time, so network jitter doesn't produce gaps.
 *
 * Interruptions are the server's call. It hears clean audio, so when someone
 * talks over the agent it says so, and this client drops every scheduled chunk
 * immediately — including the tail of a sentence already synthesized. The
 * server is told while the speaker is actually audible, because playback
 * outlasts generation and people interrupt what they hear.
 *
 * No API key ever belongs in a page. Pass a short-lived `token` instead; it
 * travels as a query parameter, which is what browsers allow on a WebSocket.
 */
(function (global) {
  "use strict";

  var VERSION = "0.1";
  var WS_PATH = "/v1/voice/ws";

  // Where this file was served from — the server to talk to, unless told otherwise.
  var SCRIPT_ORIGIN = (function () {
    try {
      var el = document.currentScript;
      return el && el.src ? new URL(el.src).origin : null;
    } catch (e) {
      return null;
    }
  })();

  /**
   * Runs inside the audio thread, not on the page.
   *
   * It resamples the microphone to the rate the server asked for and hands up
   * whole chunks, so the main thread only forwards bytes. Shipped by turning
   * this function back into source and loading it as a worklet module, which
   * is what lets one file work when a page loads it from another origin.
   */
  function captureProcessor() {
    class FusionCapture extends AudioWorkletProcessor {
      constructor(options) {
        super();
        var settings = options.processorOptions || {};
        this.ratio = sampleRate / settings.targetRate;  // sampleRate: the audio thread's own rate
        this.size = Math.max(1, Math.round(settings.targetRate * settings.chunkMs / 1000));
        this.chunk = new Int16Array(this.size);
        this.filled = 0;
        this.sum = 0;
        this.count = 0;
        this.credit = 0;
        this.last = 0;
        this.energy = 0;
      }

      push(value) {
        if (value > 1) value = 1; else if (value < -1) value = -1;
        this.energy += value * value;
        this.chunk[this.filled++] = value < 0 ? value * 32768 : value * 32767;
        if (this.filled === this.size) {
          var level = Math.sqrt(this.energy / this.size);
          var full = this.chunk.slice();
          this.filled = 0;
          this.energy = 0;
          this.port.postMessage({ pcm: full.buffer, level: level }, [full.buffer]);
        }
      }

      process(inputs) {
        var input = inputs[0];
        if (!input || !input.length || !input[0]) return true;
        var samples = input[0];
        // A box filter: each output sample is the average of the input samples
        // that fall in its window. Cheap, and it doesn't alias the way plain
        // "take every third sample" does. Ratios below 1 hold the last value.
        for (var i = 0; i < samples.length; i++) {
          this.sum += samples[i];
          this.count++;
          this.credit += 1;
          while (this.credit >= this.ratio) {
            this.credit -= this.ratio;
            if (this.count) {
              this.last = this.sum / this.count;
              this.sum = 0;
              this.count = 0;
            }
            this.push(this.last);
          }
        }
        return true;
      }
    }
    registerProcessor("fusion-capture", FusionCapture);
  }

  var CAPTURE_WORKLET = "(" + captureProcessor.toString() + ")();";

  function Emitter() {
    this._listeners = {};
  }
  Emitter.prototype.on = function (type, fn) {
    (this._listeners[type] = this._listeners[type] || []).push(fn);
    return this;
  };
  Emitter.prototype.emit = function (type, payload) {
    var fns = this._listeners[type] || [];
    for (var i = 0; i < fns.length; i++) {
      try {
        fns[i](payload);
      } catch (e) {
        console.error("[fusion-runtime] listener for " + type + " failed", e);
      }
    }
  };

  function socketUrl(options, token) {
    var url = options.url;
    if (!url) {
      url = (SCRIPT_ORIGIN || global.location.origin) + WS_PATH;
    }
    url = url.replace(/^http/, "ws");
    if (token) {
      url += (url.indexOf("?") === -1 ? "?" : "&") + "token=" + encodeURIComponent(token);
    }
    return url;
  }

  /**
   * One conversation: microphone in, replies out.
   *
   * session.start()  ask for the microphone and connect
   * session.stop()   hang up
   * session.on(type, fn) where type is one of:
   *   state ("idle" | "connecting" | "live" | "closed" | "error")
   *   level (0..1 microphone loudness, ~25x a second)
   *   transcript, response, interrupted, turn_resumed, echo_discarded, trace, error
   *   message (every server message, raw)
   */
  function Session(options) {
    Emitter.call(this);
    this.options = options || {};
    // Spent when it is used, so the server sends a replacement on connect;
    // otherwise a second "talk" would be refused.
    this.token = this.options.token || null;
    this.state = "idle";
    this.inputRate = 16000;   // the server tells us for sure in its first message
    this.outputRate = 24000;
    this.jitterMs = this.options.jitterMs || 80;
    this.chunkMs = this.options.chunkMs || 40;
    this._sources = new Set();
    this._nextPlayTime = 0;
    this._playing = false;
    this._discarding = false;
  }
  Session.prototype = Object.create(Emitter.prototype);
  Session.prototype.constructor = Session;

  Session.prototype._setState = function (state, detail) {
    this.state = state;
    this.emit("state", { state: state, detail: detail || null });
  };

  Session.prototype._fail = function (message, error) {
    this._setState("error", message);
    this.emit("error", { message: message, error: error || null });
    this.stop();
  };

  Session.prototype.start = function () {
    var self = this;
    if (this.state === "connecting" || this.state === "live") return Promise.resolve(this);
    this._setState("connecting");
    if (!global.isSecureContext) {
      // Browsers only hand over a microphone on https:// or localhost.
      this._fail("This page isn't on a secure origin, so the browser won't allow the microphone. "
                 + "Use https:// (and wss://), or open it on localhost.");
      return Promise.reject(new Error("insecure context"));
    }
    if (!global.AudioWorklet || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this._fail("This browser is too old for live audio. Chrome, Edge, Firefox or Safari 14.1+ work.");
      return Promise.reject(new Error("unsupported browser"));
    }
    return this._openMicrophone()
      .then(function () { return self._openSocket(); })
      .then(function () { self._setState("live"); return self; })
      .catch(function (e) {
        if (self.state !== "error") {
          var message = (e && e.name === "NotAllowedError")
            ? "The microphone was blocked. Allow it in the browser's address bar and try again."
            : "Couldn't start: " + ((e && e.message) || e);
          self._fail(message, e);
        }
        throw e;
      });
  };

  Session.prototype._openMicrophone = function () {
    var self = this;
    var Ctx = global.AudioContext || global.webkitAudioContext;
    return navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,   // the browser removes the agent's voice from the microphone
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1
      }
    }).then(function (stream) {
      self._stream = stream;
      // Asking for the server's rate lets the browser do the resampling properly;
      // where it can't, the worklet does it instead.
      self._captureCtx = new Ctx({ sampleRate: self.inputRate });
      self._playbackCtx = new Ctx();
      var blob = new Blob([CAPTURE_WORKLET], { type: "application/javascript" });
      var url = URL.createObjectURL(blob);
      return self._captureCtx.audioWorklet.addModule(url).then(function () {
        URL.revokeObjectURL(url);
        var node = new AudioWorkletNode(self._captureCtx, "fusion-capture", {
          processorOptions: { targetRate: self.inputRate, chunkMs: self.chunkMs }
        });
        node.port.onmessage = function (event) { self._onCapture(event.data); };
        // A worklet only runs while it's connected to something. Silence, so the
        // microphone isn't played back into the room.
        var silence = self._captureCtx.createGain();
        silence.gain.value = 0;
        self._captureCtx.createMediaStreamSource(stream).connect(node);
        node.connect(silence).connect(self._captureCtx.destination);
        self._capture = node;
        return Promise.all([self._captureCtx.resume(), self._playbackCtx.resume()]);
      });
    });
  };

  Session.prototype._openSocket = function () {
    var self = this;
    return new Promise(function (resolve, reject) {
      var ws = new WebSocket(socketUrl(self.options, self.token));
      ws.binaryType = "arraybuffer";
      self._ws = ws;
      ws.onopen = function () { resolve(); };
      ws.onerror = function () {
        reject(new Error("couldn't reach " + socketUrl(self.options, null)));
      };
      ws.onclose = function (event) {
        var why = self._refusal || event.reason || ("code " + event.code);
        if (self.state === "live") self._setState("closed", why);
        // A socket refused before it went live never reached "live", so the
        // reason would otherwise be dropped entirely.
        else if (self._refusal) self.emit("error", { message: self._refusal });
        self._refusal = null;
        self.stop();
      };
      ws.onmessage = function (event) {
        if (typeof event.data === "string") self._onServerMessage(JSON.parse(event.data));
        else self._playChunk(event.data);
      };
    });
  };

  Session.prototype._onCapture = function (data) {
    this.emit("level", data.level);
    if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
    // If the connection is backed up, drop rather than pile on latency.
    if (this._ws.bufferedAmount > 16 * this.inputRate) return;
    this._ws.send(data.pcm);
  };

  Session.prototype._send = function (message) {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) this._ws.send(JSON.stringify(message));
  };

  Session.prototype._onServerMessage = function (msg) {
    this.emit("message", msg);
    switch (msg.type) {
      case "config":
        this.sessionId = msg.session_id;
        if (msg.output_sample_rate) this.outputRate = msg.output_sample_rate;
        if (msg.next_token) this.token = msg.next_token;  // for the next connection
        break;
      case "transcript":
        if (msg.is_final) this._discarding = false;  // a new turn: play the agent again
        this.emit("transcript", msg);
        break;
      case "response":
        this.emit("response", msg);
        break;
      case "interrupted":
        // Drop what's queued, and what's still arriving, until the next turn.
        this._discarding = true;
        this.flush();
        this.emit("interrupted", msg);
        break;
      case "turn.trace":
        this.emit("trace", msg);
        break;
      case "error":
        // Kept so onclose can use it. The server refuses a socket by accepting it,
        // saying why, and then closing — without this the close handler overwrites
        // "this server needs a token" with a bare "code 1008".
        this._refusal = msg.fix ? msg.message + " " + msg.fix : msg.message;
        this.emit("error", { message: msg.message, server: msg });
        break;
      default:
        this.emit(msg.type, msg);
    }
  };

  Session.prototype._playChunk = function (arrayBuffer) {
    if (this._discarding || !this._playbackCtx) return;
    var pcm = new Int16Array(arrayBuffer);
    if (!pcm.length) return;
    var buffer = this._playbackCtx.createBuffer(1, pcm.length, this.outputRate);
    var channel = buffer.getChannelData(0);
    for (var i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
    var source = this._playbackCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(this._playbackCtx.destination);
    var now = this._playbackCtx.currentTime;
    if (this._nextPlayTime < now + 0.005) this._nextPlayTime = now + this.jitterMs / 1000;
    source.start(this._nextPlayTime);
    this._nextPlayTime += buffer.duration;
    var self = this;
    this._sources.add(source);
    source.onended = function () {
      self._sources.delete(source);
      self._reportPlayback();
    };
    this._reportPlayback();
  };

  /** Stop the agent's voice now, and forget what was queued. */
  Session.prototype.flush = function () {
    this._sources.forEach(function (source) {
      try {
        source.onended = null;
        source.stop();
      } catch (e) { /* already finished */ }
    });
    this._sources.clear();
    this._nextPlayTime = 0;
    this._reportPlayback();
  };

  /** Tell the server to stop talking — for a page with its own stop button. */
  Session.prototype.interrupt = function () {
    this._discarding = true;
    this.flush();
    this._send({ type: "interrupt" });
  };

  Session.prototype._reportPlayback = function () {
    var playing = this._sources.size > 0;
    if (playing === this._playing) return;
    this._playing = playing;
    this._send({ type: "playback", playing: playing });
  };

  Session.prototype.stop = function () {
    this.flush();
    if (this._capture) { try { this._capture.port.onmessage = null; this._capture.disconnect(); } catch (e) {} }
    if (this._stream) this._stream.getTracks().forEach(function (track) { track.stop(); });
    if (this._ws) { this._ws.onclose = null; try { this._ws.close(); } catch (e) {} }
    [this._captureCtx, this._playbackCtx].forEach(function (ctx) {
      if (ctx && ctx.state !== "closed") { try { ctx.close(); } catch (e) {} }
    });
    this._capture = this._stream = this._ws = this._captureCtx = this._playbackCtx = null;
    this._playing = false;
    if (this.state === "live" || this.state === "connecting") this._setState("idle");
    return this;
  };

  /** A session, not started yet. */
  function connect(options) {
    return new Session(options || {});
  }

  /**
   * Bind a button: click to talk, click again to hang up.
   *   FusionRuntime.attach({ button: "#talk" })
   * Returns the session, so a page can listen to transcripts and replies too.
   */
  function attach(options) {
    options = options || {};
    var session = new Session(options);
    var button = typeof options.button === "string"
      ? document.querySelector(options.button)
      : options.button;
    if (!button) throw new Error("FusionRuntime.attach: no button found for " + options.button);
    var labels = options.labels || {};
    function label(state) {
      if (state === "live") return labels.live || "Stop";
      if (state === "connecting") return labels.connecting || "Connecting…";
      return labels.idle || "Talk";
    }
    session.on("state", function (event) {
      button.textContent = label(event.state);
      button.disabled = event.state === "connecting";
      button.setAttribute("data-fusion-state", event.state);
    });
    button.textContent = label("idle");
    button.addEventListener("click", function () {
      if (session.state === "live") session.stop();
      else session.start().catch(function () { /* already reported through on("error") */ });
    });
    if (options.onEvent) session.on("message", options.onEvent);
    return session;
  }

  global.FusionRuntime = {
    connect: connect, attach: attach, Session: Session, version: VERSION,
    captureProcessor: captureProcessor,  // the audio thread's code, exposed so it can be tested
  };
})(window);
