import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Webcam from "react-webcam";
import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  AlertTriangle,
  Brain,
  Camera,
  CheckCircle2,
  Gauge,
  Loader2,
  Moon,
  Phone,
  Radar,
  RefreshCw,
  ShieldCheck,
  Timer,
  Wifi,
  WifiOff,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { analyzeFrame, getHealth, getSessionStats, resetSession } from "./api";

const CAPTURE_INTERVAL_MS = 850;
const STATS_INTERVAL_MS = 2500;

const videoConstraints = {
  width: 640,
  height: 480,
  facingMode: "user",
};

const initialPrediction = {
  status: "STANDBY",
  mode: "WAITING",
  fatigue_score: 0,
  attention_score: 100,
  phone_detected: false,
  driver_found: false,
  ear: null,
  baseline_ear: null,
  calibrated: false,
  alert: null,
};

export default function App() {
  const webcamRef = useRef(null);
  const requestInFlight = useRef(false);
  const alertArmed = useRef(true);
  const [running, setRunning] = useState(false);
  const [booting, setBooting] = useState(true);
  const [apiOnline, setApiOnline] = useState(false);
  const [prediction, setPrediction] = useState(initialPrediction);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState(null);
  const [sessionStartedAt, setSessionStartedAt] = useState(Date.now());
  const [elapsed, setElapsed] = useState(0);

  const dangerActive =
    prediction.status === "DROWSY" || prediction.phone_detected || prediction.alert;

  const statusTone = useMemo(() => {
    if (prediction.status === "DROWSY") return "danger";
    if (prediction.phone_detected || prediction.status === "LOOKING DOWN") return "warn";
    if (prediction.status === "FOCUSED") return "good";
    return "idle";
  }, [prediction]);

  useEffect(() => {
    let active = true;
    getHealth()
      .then(() => {
        if (active) setApiOnline(true);
      })
      .catch(() => {
        if (active) setApiOnline(false);
      })
      .finally(() => {
        if (active) setBooting(false);
      });

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - sessionStartedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [sessionStartedAt]);

  const playWarningTone = useCallback(() => {
    if (!alertArmed.current) return;
    alertArmed.current = false;

    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;

    const context = new AudioContext();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.type = "sine";
    oscillator.frequency.value = 880;
    gain.gain.value = 0.08;
    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start();
    oscillator.stop(context.currentTime + 0.18);

    window.setTimeout(() => {
      alertArmed.current = true;
    }, 2500);
  }, []);

  const captureAndAnalyze = useCallback(async () => {
    if (!running || requestInFlight.current || !webcamRef.current) return;

    const screenshot = webcamRef.current.getScreenshot({
      width: 480,
      height: 360,
    });
    if (!screenshot) return;

    requestInFlight.current = true;
    try {
      const result = await analyzeFrame(screenshot);
      setApiOnline(true);
      setPrediction((current) => ({ ...current, ...result }));
      setHistory((current) => [
        ...current.slice(-44),
        {
          tick: new Date().toLocaleTimeString([], {
            minute: "2-digit",
            second: "2-digit",
          }),
          ear: result.ear ?? 0,
          fatigue: result.fatigue_score ?? 0,
          attention: result.attention_score ?? 0,
        },
      ]);

      if (result.status === "DROWSY" || result.phone_detected) {
        playWarningTone();
      }
    } catch (error) {
      setApiOnline(false);
    } finally {
      requestInFlight.current = false;
    }
  }, [playWarningTone, running]);

  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setInterval(captureAndAnalyze, CAPTURE_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [captureAndAnalyze, running]);

  useEffect(() => {
    if (!running) return undefined;
    const timer = window.setInterval(() => {
      getSessionStats().then(setStats).catch(() => setApiOnline(false));
    }, STATS_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [running]);

  async function handleReset() {
    await resetSession();
    setPrediction(initialPrediction);
    setHistory([]);
    setStats(null);
    setSessionStartedAt(Date.now());
    setElapsed(0);
  }

  return (
    <main className="min-h-screen overflow-hidden bg-[#080b12] text-slate-100">
      <LoadingOverlay show={booting} />
      <div className="pointer-events-none fixed inset-0 bg-[radial-gradient(circle_at_20%_10%,rgba(34,211,238,0.16),transparent_32%),radial-gradient(circle_at_82%_0%,rgba(16,185,129,0.12),transparent_28%),linear-gradient(135deg,rgba(15,23,42,0.85),rgba(2,6,23,0.96))]" />
      <div className="relative mx-auto flex min-h-screen max-w-7xl flex-col px-4 py-5 sm:px-6 lg:px-8">
        <NavBar apiOnline={apiOnline} running={running} elapsed={elapsed} />

        <section className="mt-5 grid flex-1 gap-5 lg:grid-cols-[1.35fr_0.9fr]">
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel min-h-[560px] p-4"
          >
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase text-cyan-200/70">Live perception stream</p>
                <h1 className="mt-1 text-2xl font-semibold text-white">
                  Cognitive Fatigue Monitor
                </h1>
              </div>
              <div className="flex gap-2">
                <button className="control-button" onClick={() => setRunning((value) => !value)}>
                  <Camera className="h-4 w-4" />
                  {running ? "Pause" : "Start"}
                </button>
                <button className="icon-button" onClick={handleReset} title="Reset session">
                  <RefreshCw className="h-4 w-4" />
                </button>
              </div>
            </div>

            <div className="relative overflow-hidden rounded-lg border border-cyan-300/15 bg-black shadow-neon">
              <Webcam
                ref={webcamRef}
                audio={false}
                mirrored
                screenshotFormat="image/jpeg"
                screenshotQuality={0.64}
                videoConstraints={videoConstraints}
                className="aspect-video w-full object-cover"
              />
              <div className="pointer-events-none absolute left-4 top-4 rounded-md border border-cyan-300/30 bg-black/35 px-3 py-2 backdrop-blur">
                <div className="flex items-center gap-2 text-sm text-cyan-100">
                  <Radar className="h-4 w-4" />
                  {running ? "Analyzing" : "Standby"}
                </div>
              </div>
              <AnimatePresence>
                {dangerActive && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-x-0 bottom-0 border-t border-red-300/30 bg-red-500/15 px-4 py-3 backdrop-blur"
                  >
                    <div className="flex items-center gap-2 font-semibold text-red-100">
                      <AlertTriangle className="h-5 w-5" />
                      {prediction.alert || prediction.status}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              <MetricCard
                icon={Brain}
                label="Fatigue score"
                value={prediction.fatigue_score}
                suffix="/100"
                tone={prediction.fatigue_score > 75 ? "danger" : "cyan"}
              />
              <MetricCard
                icon={ShieldCheck}
                label="Attention score"
                value={prediction.attention_score}
                suffix="/100"
                tone={prediction.attention_score < 55 ? "warn" : "green"}
              />
              <MetricCard
                icon={Gauge}
                label="EAR"
                value={prediction.ear ?? "--"}
                suffix={prediction.baseline_ear ? ` base ${prediction.baseline_ear}` : ""}
                tone="violet"
              />
            </div>
          </motion.div>

          <aside className="grid gap-5">
            <StatusPanel prediction={prediction} statusTone={statusTone} />
            <AnalyticsPanel stats={stats} prediction={prediction} />
          </aside>
        </section>

        <section className="mt-5 grid gap-5 lg:grid-cols-[1fr_0.8fr]">
          <ChartPanel history={history} />
          <SystemPanel apiOnline={apiOnline} prediction={prediction} running={running} />
        </section>
      </div>
    </main>
  );
}

function LoadingOverlay({ show }) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          className="fixed inset-0 z-50 grid place-items-center bg-[#080b12]"
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <div className="text-center">
            <Loader2 className="mx-auto h-9 w-9 animate-spin text-cyan-300" />
            <p className="mt-4 text-sm uppercase text-cyan-100/70">Booting neural runtime</p>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function NavBar({ apiOnline, running, elapsed }) {
  return (
    <nav className="glass-panel flex flex-wrap items-center justify-between gap-3 px-4 py-3">
      <div className="flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-lg border border-cyan-300/30 bg-cyan-300/10">
          <Activity className="h-5 w-5 text-cyan-200" />
        </div>
        <div>
          <p className="text-sm font-semibold text-white">NeuroDrive AI</p>
          <p className="text-xs text-slate-400">Driver attention intelligence</p>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <Pill icon={apiOnline ? Wifi : WifiOff} label={apiOnline ? "API Online" : "API Offline"} />
        <Pill icon={running ? CheckCircle2 : Moon} label={running ? "Session Active" : "Session Idle"} />
        <Pill icon={Timer} label={formatTime(elapsed)} />
      </div>
    </nav>
  );
}

function StatusPanel({ prediction, statusTone }) {
  return (
    <motion.div layout className={`glass-panel p-4 status-${statusTone}`}>
      <p className="text-xs uppercase text-slate-400">Current mode</p>
      <h2 className="mt-2 text-3xl font-semibold text-white">{prediction.mode}</h2>
      <div className="mt-4 grid grid-cols-2 gap-3">
        <StatusTile label="Status" value={prediction.status} />
        <StatusTile label="Driver" value={prediction.driver_found ? "Verified" : "Searching"} />
        <StatusTile label="Phone" value={prediction.phone_detected ? "Detected" : "Clear"} />
        <StatusTile label="Calibration" value={prediction.calibrated ? "Ready" : "Learning"} />
      </div>
    </motion.div>
  );
}

function AnalyticsPanel({ stats, prediction }) {
  return (
    <div className="glass-panel p-4">
      <p className="text-xs uppercase text-slate-400">Session analytics</p>
      <div className="mt-4 grid grid-cols-2 gap-3">
        <StatusTile label="Frames" value={stats?.frames_processed ?? 0} />
        <StatusTile label="Drowsy events" value={stats?.drowsy_events ?? 0} />
        <StatusTile label="Phone events" value={stats?.phone_events ?? 0} />
        <StatusTile label="Baseline" value={prediction.baseline_ear ?? "--"} />
      </div>
    </div>
  );
}

function ChartPanel({ history }) {
  return (
    <div className="glass-panel h-[320px] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <p className="text-xs uppercase text-slate-400">Live EAR telemetry</p>
          <h2 className="text-lg font-semibold text-white">Attention waveform</h2>
        </div>
      </div>
      <ResponsiveContainer width="100%" height="82%">
        <AreaChart data={history}>
          <defs>
            <linearGradient id="earGradient" x1="0" x2="0" y1="0" y2="1">
              <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.55} />
              <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(148, 163, 184, 0.12)" vertical={false} />
          <XAxis dataKey="tick" hide />
          <YAxis domain={[0, 1]} stroke="#64748b" width={34} />
          <Tooltip
            contentStyle={{
              background: "rgba(8, 11, 18, 0.92)",
              border: "1px solid rgba(34, 211, 238, 0.22)",
              borderRadius: 8,
              color: "#e2e8f0",
            }}
          />
          <Area type="monotone" dataKey="ear" stroke="#22d3ee" fill="url(#earGradient)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

function SystemPanel({ apiOnline, prediction, running }) {
  return (
    <div className="glass-panel p-4">
      <p className="text-xs uppercase text-slate-400">System status</p>
      <div className="mt-4 space-y-3">
        <SignalRow label="Backend API" active={apiOnline} />
        <SignalRow label="Camera stream" active={running} />
        <SignalRow label="Driver verification" active={prediction.driver_found} />
        <SignalRow label="Fatigue calibration" active={prediction.calibrated} />
      </div>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, suffix, tone }) {
  return (
    <motion.div whileHover={{ y: -2 }} className={`metric-card tone-${tone}`}>
      <div className="flex items-center justify-between">
        <p className="text-xs uppercase text-slate-400">{label}</p>
        <Icon className="h-4 w-4 text-cyan-200" />
      </div>
      <div className="mt-3 flex items-end gap-1">
        <span className="text-3xl font-semibold text-white">{value}</span>
        <span className="pb-1 text-xs text-slate-400">{suffix}</span>
      </div>
    </motion.div>
  );
}

function StatusTile({ label, value }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.04] p-3">
      <p className="text-xs text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-semibold text-white">{value}</p>
    </div>
  );
}

function SignalRow({ label, active }) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2">
      <span className="text-sm text-slate-300">{label}</span>
      <span className={`h-2.5 w-2.5 rounded-full ${active ? "bg-emerald-300" : "bg-red-300"}`} />
    </div>
  );
}

function Pill({ icon: Icon, label }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.05] px-3 py-2 text-slate-200">
      <Icon className="h-4 w-4 text-cyan-200" />
      <span>{label}</span>
    </div>
  );
}

function formatTime(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60).toString().padStart(2, "0");
  const seconds = (totalSeconds % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}
