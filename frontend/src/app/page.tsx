
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const WS = API.replace(/^http/, "ws") + "/ws/detections";

const C = {
  bg: "#E5EFF0",
  white: "#FFFFFF",
  header: "#3D5C6E",
  teal: "#66909E",
  mist: "#92B8C1",
  purple: "#776A95",
  lavender: "#B8AADD",
  text: "#263E4B",
  muted: "#718792",
  border: "#D4E2E5",
  pale: "#F2F7F7",
};

type Detection = {
  track_id?: number | null;
  class_id?: number;
  class_name: string;
  confidence: number;
  bbox?: number[];
};

type Event = {
  event_id?: string;
  source: string;
  filename?: string;
  timestamp?: string;
  frame_id?: number | null;
  video_timestamp?: number | null;
  object_count: number;
  visible_objects?: number;
  tracked_objects?: number;
  objects: Detection[];
};

type VideoInfo = {
  media_id: string;
  filename: string;
  fps: number;
  total_frames: number;
  duration_seconds: number;
  width: number;
  height: number;
};

type FrameResult = {
  frame_index: number;
  video_timestamp: number;
  detections: Detection[];
  object_count: number;
  annotated_image: string;
  activity_event: Event;
};

function clock(seconds: number) {
  if (!Number.isFinite(seconds)) return "00:00.00";

  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;

  return `${String(mins).padStart(2, "0")}:${secs.toFixed(2).padStart(5, "0")}`;
}

function localTime(value?: string) {
  if (!value) return "Time unavailable";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString();
}

export default function Home() {
  const [mode, setMode] = useState<"live" | "media">("live");
  const [switchingMode, setSwitchingMode] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);
  const [connected, setConnected] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);

  const [liveEvents, setLiveEvents] = useState<Event[]>([]);
  const [mediaEvents, setMediaEvents] = useState<Event[]>([]);
  const [lastUpdate, setLastUpdate] = useState("Waiting for data");

  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [selectedFileUrl, setSelectedFileUrl] = useState("");
  const [selectedFileType, setSelectedFileType] =
    useState<"image" | "video" | null>(null);
  const [imageResult, setImageResult] = useState("");
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null>(null);
  const [frameResult, setFrameResult] = useState<FrameResult | null>(null);
  const [frameLoading, setFrameLoading] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const frameRequestRef = useRef(0);
  const lastFrameRequestRef = useRef(0);
  const lastEventFrameRef = useRef(-1);
  const lastActivityTimeRef = useRef(0);

  // Backend health
  useEffect(() => {
    let active = true;

    const check = async () => {
      try {
        const response = await fetch(`${API}/health`);
        if (active) setBackendOnline(response.ok);
      } catch {
        if (active) setBackendOnline(false);
      }
    };

    check();
    const timer = setInterval(check, 5000);

    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  // Live detection WebSocket
  useEffect(() => {
    let active = true;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connect = () => {
      if (!active) return;

      socket = new WebSocket(WS);

      socket.onopen = () => {
        if (active) setConnected(true);
      };

      socket.onmessage = (message) => {
        try {
          const payload = JSON.parse(message.data);

          if (payload.type !== "detection" || !payload.data) return;

          const raw = payload.data;

          const event: Event = {
            ...raw,
            source: "live_camera",
            timestamp: raw.timestamp || new Date().toISOString(),
            objects: raw.objects || [],
            object_count: raw.object_count || 0,
          };

          if (active) {
            setLiveEvents((previous) => [event, ...previous].slice(0, 200));
            setLastUpdate(new Date().toLocaleTimeString());
            setCameraActive(true);
          }
        } catch {
          // Ignore malformed event messages.
        }
      };

      socket.onclose = () => {
        if (active) {
          setConnected(false);
          reconnectTimer = setTimeout(connect, 3000);
        }
      };

      socket.onerror = () => socket?.close();
    };

    connect();

    return () => {
      active = false;
      clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  // Release browser preview URL on replacement/unmount
  useEffect(() => {
    return () => {
      if (selectedFileUrl.startsWith("blob:")) {
        URL.revokeObjectURL(selectedFileUrl);
      }
    };
  }, [selectedFileUrl]);

  // Real backend mode switch
  const changeMode = async (nextMode: "live" | "media") => {
    if (nextMode === mode || switchingMode) return;

    setSwitchingMode(true);
    setUploadError("");

    try {
      const response = await fetch(`${API}/mode`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: nextMode }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Could not change operating mode");
      }

      if (nextMode === "media") {
        setCameraActive(false);
      }

      setMode(nextMode);
    } catch (error) {
      setUploadError(
        error instanceof Error ? error.message : "Mode change failed"
      );
    } finally {
      setSwitchingMode(false);
    }
  };

  // Upload image or video
  const uploadMedia = async (file: File) => {
    const extension = file.name.split(".").pop()?.toLowerCase() || "";
    const isImage = ["jpg", "jpeg", "png", "webp"].includes(extension);
    const isVideo = extension === "mp4";

    if (!isImage && !isVideo) {
      setUploadError("Upload JPG, PNG, WEBP or MP4 only.");
      return;
    }

    const limit = isImage ? 10 * 1024 * 1024 : 50 * 1024 * 1024;

    if (file.size > limit) {
      setUploadError(`Maximum file size is ${limit / 1024 / 1024} MB.`);
      return;
    }

    if (selectedFileUrl.startsWith("blob:")) {
      URL.revokeObjectURL(selectedFileUrl);
    }

    setSelectedFileUrl(URL.createObjectURL(file));
    setSelectedFileType(isImage ? "image" : "video");
    setImageResult("");
    setVideoInfo(null);
    setFrameResult(null);
    setMediaEvents([]);
    setUploadError("");
    setUploading(true);

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${API}/media/upload`, {
        method: "POST",
        body: form,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Upload failed");
      }

      if (result.media_type === "image") {
        setImageResult(result.annotated_image);

        setMediaEvents(result.activity_events || []);
        setLastUpdate(new Date().toLocaleTimeString());
      } else {
        setVideoInfo({
          media_id: result.media_id,
          filename: result.filename,
          fps: result.fps,
          total_frames: result.total_frames,
          duration_seconds: result.duration_seconds,
          width: result.width,
          height: result.height,
        });

        lastFrameRequestRef.current = 0;
        lastEventFrameRef.current = -1;
      }
    } catch (error) {
      setUploadError(
        error instanceof Error ? error.message : "Upload failed"
      );
    } finally {
      setUploading(false);
    }
  };

  // Request exact selected video frame from backend
  const analyzeFrame = useCallback(
    async (frameIndex: number, force = false) => {
      if (!videoInfo) return;

      const boundedFrame = Math.max(
        0,
        Math.min(videoInfo.total_frames - 1, frameIndex)
      );

      if (!force && boundedFrame === lastFrameRequestRef.current) return;

      lastFrameRequestRef.current = boundedFrame;

      const requestId = ++frameRequestRef.current;
      setFrameLoading(true);

      try {
        const url =
          `${API}/media/${videoInfo.media_id}/frame` +
          `?frame_index=${boundedFrame}`;

        const response = await fetch(url);

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || "Frame analysis failed");
        }

        const result: FrameResult = await response.json();

        // Ignore stale results if the user has already moved to another frame.
        if (requestId !== frameRequestRef.current) return;

        setFrameResult(result);
        setLastUpdate(new Date().toLocaleTimeString());

        // Record activity approximately once per second during playback,
        // and always when paused/seeking/stepping.
        const now = Date.now();
        const video = videoRef.current;
        const paused = !video || video.paused;

        if (
          paused ||
          now - lastActivityTimeRef.current >= 1000 ||
          boundedFrame === 0
        ) {
          setMediaEvents((previous) => {
            const filtered = previous.filter(
              (event) => event.frame_id !== boundedFrame
            );

            return [
              result.activity_event,
              ...filtered,
            ].slice(0, 200);
          });

          lastActivityTimeRef.current = now;
        }
      } catch (error) {
        if (requestId === frameRequestRef.current) {
          setUploadError(
            error instanceof Error ? error.message : "Frame analysis failed"
          );
        }
      } finally {
        if (requestId === frameRequestRef.current) {
          setFrameLoading(false);
        }
      }
    },
    [videoInfo]
  );

  // Map video playback position to exact frame number.
  const handleVideoTime = (force = false) => {
    const video = videoRef.current;

    if (!video || !videoInfo || !Number.isFinite(video.currentTime)) return;

    const frameIndex = Math.min(
      videoInfo.total_frames - 1,
      Math.max(0, Math.floor(video.currentTime * videoInfo.fps + 0.00001))
    );

    const now = Date.now();

    // During playback, request frames at most every 250 ms.
    if (
      !force &&
      !video.paused &&
      now - lastActivityTimeRef.current < 250
    ) {
      return;
    }

    analyzeFrame(frameIndex, force);
  };

  const seekBy = (seconds: number) => {
    const video = videoRef.current;
    if (!video || !videoInfo) return;

    video.currentTime = Math.max(
      0,
      Math.min(videoInfo.duration_seconds, video.currentTime + seconds)
    );

    handleVideoTime(true);
  };

  const stepFrame = (direction: number) => {
    const video = videoRef.current;
    if (!video || !videoInfo) return;

    video.pause();

    const nextTime =
      video.currentTime + direction / videoInfo.fps;

    video.currentTime = Math.max(
      0,
      Math.min(
        videoInfo.duration_seconds,
        nextTime
      )
    );

    // The seeked event also requests the exact frame.
    setTimeout(() => handleVideoTime(true), 50);
  };

  const visibleEvents = mode === "live" ? liveEvents : mediaEvents;

  const latestLive = liveEvents[0];
  const latestMedia = mediaEvents[0];

  const latestObjects =
    mode === "live"
      ? latestLive?.objects || []
      : latestMedia?.objects || [];

  const visibleCount =
    mode === "live"
      ? latestLive?.visible_objects ?? latestLive?.object_count ?? 0
      : latestMedia?.object_count ?? 0;

  const trackedCount =
    mode === "live"
      ? latestLive?.tracked_objects ?? latestLive?.object_count ?? 0
      : latestMedia?.object_count ?? 0;

  const averageConfidence = latestObjects.length
    ? latestObjects.reduce((sum, item) => sum + item.confidence, 0) /
      latestObjects.length * 100
    : 0;

  const categories = useMemo(() => {
    const counts: Record<string, number> = {};

    latestObjects.forEach((object) => {
      counts[object.class_name] = (counts[object.class_name] || 0) + 1;
    });

    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [latestObjects]);

  return (
    <main className="min-h-screen" style={{ background: C.bg, color: C.text }}>
      <header className="sticky top-0 z-20 shadow-sm" style={{ background: C.header }}>
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-3">
            <div
              className="flex h-11 w-11 items-center justify-center rounded-xl text-2xl"
              style={{ background: "#FFFFFF20", color: C.lavender }}
            >
              â—ˆ
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">
                Aero<span style={{ color: C.lavender }}>Vision</span>
              </h1>
              <p className="text-xs" style={{ color: C.mist }}>
                Aerial intelligence & visual analytics
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <Status label="API" active={backendOnline} />
            <Status label="WebSocket" active={connected} />
            <Status label="Camera" active={cameraActive && mode === "live"} />
            <span className="rounded-lg border border-white/20 px-3 py-2 text-white">
              {mode === "live" ? "LIVE MONITORING" : "MEDIA ANALYSIS"}
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1600px] space-y-7 px-6 py-8">
        <section className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold">Vision Dashboard</h2>
            <p className="mt-2 text-sm" style={{ color: C.muted }}>
              {mode === "live"
                ? "Live camera monitoring and object recognition"
                : "Uploaded media analysis and frame-by-frame inspection"}
            </p>
          </div>

          <div className="rounded-xl border bg-white px-5 py-3" style={{ borderColor: C.border }}>
            <p className="text-[10px] font-bold tracking-widest" style={{ color: C.muted }}>
              LAST ANALYSIS UPDATE
            </p>
            <p className="mt-1 text-sm font-semibold">{lastUpdate}</p>
          </div>
        </section>

        {/* Mode selector */}
        <section className="rounded-2xl border bg-white p-2" style={{ borderColor: C.border }}>
          <div className="grid grid-cols-2 gap-2">
            <button
              disabled={switchingMode}
              onClick={() => changeMode("live")}
              className="rounded-xl px-4 py-4 text-sm font-bold disabled:opacity-50"
              style={{
                background: mode === "live" ? C.header : C.pale,
                color: mode === "live" ? C.white : C.text,
              }}
            >
              â—‰ Live Monitoring
            </button>

            <button
              disabled={switchingMode}
              onClick={() => changeMode("media")}
              className="rounded-xl px-4 py-4 text-sm font-bold disabled:opacity-50"
              style={{
                background: mode === "media" ? C.purple : C.pale,
                color: mode === "media" ? C.white : C.purple,
              }}
            >
              â‡§ Media Analysis
            </button>
          </div>

          {switchingMode && (
            <p className="py-2 text-center text-xs" style={{ color: C.muted }}>
              Switching operating mode...
            </p>
          )}
        </section>

        {/* Mode-specific statistics */}
        <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Stat label="DETECTION EVENTS" value={visibleEvents.length} description="Current mode history" />
          <Stat label="OBJECTS IN CURRENT FRAME" value={visibleCount} description="Selected frame detections" />
          <Stat label="TRACKED OBJECTS" value={trackedCount} description="Current mode" />
          <Stat label="AVG. CONFIDENCE" value={`${averageConfidence.toFixed(1)}%`} description="Latest analysed frame" />
        </section>

        {uploadError && (
          <div className="rounded-xl bg-red-50 p-4 text-sm text-red-700">
            {uploadError}
            <button
              className="ml-3 font-bold underline"
              onClick={() => setUploadError("")}
            >
              Dismiss
            </button>
          </div>
        )}

        {/* LIVE MODE */}
        {mode === "live" && (
          <section className="space-y-4">
            <div>
              <h3 className="text-lg font-bold">Live Camera Monitoring</h3>
              <p className="text-xs" style={{ color: C.muted }}>
                Live original and AI-annotated camera streams
              </p>
            </div>

            <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
              <VideoPanel
                title="RAW CAMERA FEED"
                subtitle="Original camera stream"
                endpoint={`${API}/video/raw`}
                accent={C.teal}
              />
              <VideoPanel
                title="AI DETECTION FEED"
                subtitle="YOLOv5 annotated stream"
                endpoint={`${API}/video/annotated`}
                accent={C.purple}
              />
            </div>
          </section>
        )}

        {/* MEDIA MODE */}
        {mode === "media" && (
          <section className="space-y-5">
            <div>
              <h3 className="text-lg font-bold">Media Analysis Workspace</h3>
              <p className="text-xs" style={{ color: C.muted }}>
                Upload one image or MP4 video. Analyse individual frames.
              </p>
            </div>

            <div className="rounded-2xl border bg-white p-5" style={{ borderColor: C.border }}>
              <button
                onClick={() => inputRef.current?.click()}
                disabled={uploading}
                className="flex min-h-[150px] w-full flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 disabled:opacity-50"
                style={{ borderColor: C.mist, background: C.pale }}
              >
                <span className="text-3xl" style={{ color: C.teal }}>â‡§</span>
                <span className="mt-2 font-semibold">
                  {uploading ? "Uploading media..." : "Choose image or MP4 video"}
                </span>
                <span className="mt-1 text-xs" style={{ color: C.muted }}>
                  JPG Â· PNG Â· WEBP Â· MP4
                </span>
                <span className="mt-1 text-xs" style={{ color: C.muted }}>
                  Images up to 10 MB Â· Videos up to 50 MB
                </span>
              </button>

              <input
                ref={inputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.webp,.mp4"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) uploadMedia(file);
                  event.target.value = "";
                }}
              />

              {uploading && (
                <p className="mt-4 text-center text-sm" style={{ color: C.teal }}>
                  Preparing your media...
                </p>
              )}

              {/* IMAGE WORKSPACE */}
              {selectedFileType === "image" && selectedFileUrl && (
                <div className="mt-6 grid grid-cols-1 gap-5 xl:grid-cols-2">
                  <MediaPanel title="ORIGINAL IMAGE">
                    <img src={selectedFileUrl} className="max-h-[500px] w-full object-contain" alt="Original upload" />
                  </MediaPanel>

                  <MediaPanel title="AI ANNOTATED IMAGE">
                    {imageResult ? (
                      <img src={imageResult} className="max-h-[500px] w-full object-contain" alt="Annotated image" />
                    ) : (
                      <Empty message="Waiting for image analysis..." />
                    )}
                  </MediaPanel>
                </div>
              )}

              {/* VIDEO WORKSPACE */}
              {selectedFileType === "video" && selectedFileUrl && videoInfo && (
                <div className="mt-6 space-y-5">
                  <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
                    <MediaPanel title="ORIGINAL VIDEO">
                      <video
                        ref={videoRef}
                        src={selectedFileUrl}
                        controls
                        playsInline
                        className="max-h-[450px] w-full bg-black"
                        onLoadedMetadata={() => {
                          const video = videoRef.current;
                          if (video) {
                            video.currentTime = 0;
                            analyzeFrame(0, true);
                          }
                        }}
                        onTimeUpdate={() => handleVideoTime(false)}
                        onSeeked={() => handleVideoTime(true)}
                        onPause={() => handleVideoTime(true)}
                      />
                    </MediaPanel>

                    <MediaPanel title="SYNCHRONIZED AI FRAME">
                      {frameResult?.annotated_image ? (
                        <img
                          src={frameResult.annotated_image}
                          alt="AI annotated selected frame"
                          className="max-h-[450px] w-full bg-black object-contain"
                        />
                      ) : (
                        <div className="flex min-h-[240px] items-center justify-center">
                          <p className="text-sm" style={{ color: C.muted }}>
                            {frameLoading ? "Analysing frame..." : "Waiting for frame"}
                          </p>
                        </div>
                      )}
                    </MediaPanel>
                  </div>

                  {/* Shared playback controls */}
                  <div className="rounded-xl border p-4" style={{ borderColor: C.border, background: C.pale }}>
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                      <span className="text-sm font-bold">Frame controls</span>
                      <span className="text-xs" style={{ color: C.muted }}>
                        {videoInfo.fps.toFixed(2)} FPS Â· {videoInfo.total_frames} frames
                      </span>
                    </div>

                    <input
                      type="range"
                      min={0}
                      max={Math.max(0, videoInfo.duration_seconds)}
                      step={1 / videoInfo.fps}
                      value={videoRef.current?.currentTime || 0}
                      onChange={(event) => {
                        const video = videoRef.current;
                        if (!video) return;
                        video.pause();
                        video.currentTime = Number(event.target.value);
                        handleVideoTime(true);
                      }}
                      className="w-full accent-[#66909E]"
                    />

                    <div className="mt-2 flex justify-between text-xs" style={{ color: C.muted }}>
                      <span>{clock(videoRef.current?.currentTime || 0)}</span>
                      <span>{clock(videoInfo.duration_seconds)}</span>
                    </div>

                    <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
                      <ControlButton label="âˆ’5s" onClick={() => seekBy(-5)} />
                      <ControlButton label="âˆ’1s" onClick={() => seekBy(-1)} />
                      <ControlButton label="âˆ’1 frame" onClick={() => stepFrame(-1)} />
                      <ControlButton
                        label={videoRef.current?.paused === false ? "Pause" : "Play"}
                        primary
                        onClick={() => {
                          const video = videoRef.current;
                          if (!video) return;
                          if (video.paused) {
                            video.play();
                          } else {
                            video.pause();
                          }
                        }}
                      />
                      <ControlButton label="+1 frame" onClick={() => stepFrame(1)} />
                      <ControlButton label="+1s" onClick={() => seekBy(1)} />
                      <ControlButton label="+5s" onClick={() => seekBy(5)} />
                    </div>

                    <div className="mt-4 flex flex-wrap justify-between gap-2 text-xs" style={{ color: C.muted }}>
                      <span>
                        Frame: {frameResult?.frame_index ?? 0}
                      </span>
                      <span>
                        Video time: {clock(frameResult?.video_timestamp ?? 0)}
                      </span>
                      <span>
                        {frameLoading ? "Analysing..." : "Frame analysis ready"}
                      </span>
                    </div>
                  </div>

                  {/* Selected frame detections */}
                  <div className="rounded-xl border bg-white p-4" style={{ borderColor: C.border }}>
                    <div className="flex flex-wrap justify-between gap-2">
                      <h4 className="font-bold">Selected Frame Analysis</h4>
                      <span className="text-xs" style={{ color: C.muted }}>
                        Frame {frameResult?.frame_index ?? "â€”"} Â· {clock(frameResult?.video_timestamp ?? 0)}
                      </span>
                    </div>

                    <DetectionList detections={frameResult?.detections || []} />
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {/* Activity is filtered by the selected operating mode */}
        <section className="grid grid-cols-1 gap-5 lg:grid-cols-3">
          <div className="rounded-2xl border bg-white p-5 shadow-sm lg:col-span-2" style={{ borderColor: C.border }}>
            <div className="mb-5 flex items-center justify-between gap-3">
              <div>
                <h3 className="font-bold">Detection Activity</h3>
                <p className="mt-1 text-xs" style={{ color: C.muted }}>
                  {mode === "live" ? "Live camera detections" : "Uploaded media detections"}
                </p>
              </div>
              <span className="rounded-lg px-3 py-2 text-xs font-bold" style={{ background: "#EDE8F5", color: C.purple }}>
                {visibleEvents.length} EVENTS
              </span>
            </div>

            <div className="max-h-[500px] space-y-3 overflow-y-auto">
              {visibleEvents.length === 0 ? (
                <Empty message="No detection activity yet." />
              ) : (
                visibleEvents.map((event, index) => (
                  <div key={event.event_id || index} className="rounded-xl border p-4" style={{ borderColor: C.border, background: C.pale }}>
                    <div className="flex flex-wrap justify-between gap-2">
                      <div>
                        <p className="text-sm font-bold">
                          {event.filename || (event.frame_id != null ? `Frame ${event.frame_id}` : "Live camera")}
                        </p>
                        <p className="mt-1 text-xs" style={{ color: C.muted }}>
                          Detected at: {localTime(event.timestamp)}
                        </p>
                        {event.frame_id != null && (
                          <p className="mt-1 text-xs" style={{ color: C.muted }}>
                            Frame: {event.frame_id}
                          </p>
                        )}
                        {event.video_timestamp != null && (
                          <p className="mt-1 text-xs font-semibold" style={{ color: C.purple }}>
                            Video time: {clock(event.video_timestamp)}
                          </p>
                        )}
                      </div>
                      <span className="h-fit rounded-md px-2 py-1 text-[10px] font-bold" style={{ background: "#EDE8F5", color: C.purple }}>
                        {event.source.replaceAll("_", " ").toUpperCase()}
                      </span>
                    </div>
                    <DetectionList detections={event.objects || []} compact />
                    <p className="mt-3 text-xs font-semibold" style={{ color: C.teal }}>
                      {event.object_count} objects detected
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="rounded-2xl border bg-white p-5 shadow-sm" style={{ borderColor: C.border }}>
            <h3 className="font-bold">Object Categories</h3>
            <p className="mb-6 mt-1 text-xs" style={{ color: C.muted }}>
              {mode === "live" ? "Latest live frame" : "Selected media frame"}
            </p>

            {categories.length === 0 ? (
              <Empty message="No objects detected." />
            ) : (
              <div className="space-y-5">
                {categories.map(([name, count]) => (
                  <div key={name}>
                    <div className="mb-2 flex justify-between text-sm">
                      <span className="capitalize">{name}</span>
                      <strong style={{ color: C.purple }}>{count}</strong>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full" style={{ background: "#E7EFF0" }}>
                      <div className="h-full rounded-full" style={{ width: `${count / categories[0][1] * 100}%`, background: C.teal }} />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>

        <footer className="flex flex-wrap justify-between gap-3 border-t py-5 text-xs" style={{ borderColor: C.border, color: C.muted }}>
          <span className="font-semibold">AeroVision Â· Aerial Intelligence Platform</span>
          <span>YOLOv5n Â· ONNX Runtime Â· FastAPI Â· Next.js</span>
        </footer>
      </div>
    </main>
  );
}

function Status({ label, active }: { label: string; active: boolean }) {
  return (
    <span className="flex items-center gap-2 rounded-lg border border-white/20 px-3 py-2">
      <span className={`h-2 w-2 rounded-full ${active ? "bg-emerald-400" : "bg-red-400"}`} />
      <span className="text-white">{label}</span>
      <span className={active ? "text-emerald-300" : "text-red-300"}>
        {active ? "ONLINE" : "OFFLINE"}
      </span>
    </span>
  );
}

function Stat({ label, value, description }: { label: string; value: string | number; description: string }) {
  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm" style={{ borderColor: C.border }}>
      <p className="text-[11px] font-bold tracking-wider" style={{ color: C.muted }}>{label}</p>
      <p className="mt-4 text-3xl font-bold tabular-nums">{value}</p>
      <p className="mt-2 text-xs" style={{ color: C.muted }}>{description}</p>
    </div>
  );
}

function VideoPanel({ title, subtitle, endpoint, accent }: { title: string; subtitle: string; endpoint: string; accent: string }) {
  const [failed, setFailed] = useState(false);

  return (
    <div className="overflow-hidden rounded-2xl border bg-white shadow-sm" style={{ borderColor: C.border }}>
      <div className="flex items-center justify-between border-b px-5 py-4" style={{ borderColor: C.border }}>
        <div>
          <h4 className="text-sm font-bold">{title}</h4>
          <p className="mt-1 text-xs" style={{ color: C.muted }}>{subtitle}</p>
        </div>
        <span className="rounded-md px-3 py-1.5 text-xs font-bold" style={{ background: `${accent}20`, color: accent }}>LIVE</span>
      </div>
      <div className="aspect-[4/3] bg-[#18232B]">
        {failed ? (
          <div className="flex h-full items-center justify-center p-5 text-sm text-white">
            Camera stream unavailable
          </div>
        ) : (
          <img src={endpoint} alt={title} className="h-full w-full object-contain" onError={() => setFailed(true)} />
        )}
      </div>
    </div>
  );
}

function MediaPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="overflow-hidden rounded-xl border bg-white" style={{ borderColor: C.border }}>
      <div className="border-b px-4 py-3 text-sm font-bold" style={{ borderColor: C.border }}>
        {title}
      </div>
      <div className="flex min-h-[240px] items-center justify-center bg-[#18232B]">
        {children}
      </div>
    </div>
  );
}

function ControlButton({ label, onClick, primary = false }: { label: string; onClick: () => void; primary?: boolean }) {
  return (
    <button
      onClick={onClick}
      className="rounded-lg px-3 py-2 text-xs font-bold transition hover:opacity-80"
      style={{
        background: primary ? C.header : C.white,
        color: primary ? C.white : C.text,
        border: `1px solid ${C.border}`,
      }}
    >
      {label}
    </button>
  );
}

function DetectionList({ detections, compact = false }: { detections: Detection[]; compact?: boolean }) {
  if (!detections.length) {
    return <p className="mt-3 text-xs" style={{ color: C.muted }}>No objects detected in this frame.</p>;
  }

  return (
    <div className={`mt-3 flex flex-wrap gap-2 ${compact ? "" : "mt-4"}`}>
      {detections.map((item, index) => (
        <span key={`${item.class_name}-${index}`} className="rounded-md px-2 py-1 text-xs font-medium" style={{ background: "#DCEBED", color: C.text }}>
          {item.track_id != null ? `ID ${item.track_id} Â· ` : ""}
          {item.class_name} Â· {(item.confidence * 100).toFixed(1)}%
        </span>
      ))}
    </div>
  );
}

function Empty({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-dashed px-4 py-10 text-center text-sm" style={{ borderColor: C.border, background: C.pale, color: C.muted }}>
      {message}
    </div>
  );
}

