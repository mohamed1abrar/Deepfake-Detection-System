import { useEffect, useRef } from "react";

export default function VideoOverlay({ videoUrl, frames }) {
  const videoRef = useRef();
  const canvasRef = useRef();

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    function draw() {
      if (!video || video.paused || video.ended) return;

      const t = video.currentTime;
      const fps = 25;
      const frameIndex = Math.floor(t * fps);

      const frame = frames.find(f => f.frame === frameIndex);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (frame) {
        const [x1, y1, x2, y2] = frame.bbox;
        const color = frame.label === "REAL" ? "#22c55e" : "#ef4444";

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        ctx.fillStyle = color;
        ctx.font = "16px sans-serif";
        ctx.fillText(
          `${frame.label} ${(frame.avg * 100).toFixed(1)}%`,
          x1,
          y1 - 8
        );
      }

      requestAnimationFrame(draw);
    }

    video?.addEventListener("play", draw);
    return () => video?.removeEventListener("play", draw);
  }, [frames]);

  return (
    <div className="video-wrapper">
      <video ref={videoRef} src={videoUrl} controls />
      <canvas ref={canvasRef} width={640} height={360} />
    </div>
  );
}
