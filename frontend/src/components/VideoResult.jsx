const API_BASE =
  import.meta.env.VITE_API_URL ||
  "https://deepfake-detection-system-vdxr.onrender.com";

export default function VideoResult({ data }) {
  if (!data) return null;

  return (
    <>
      <video
        controls
        width="600"
        src={`${API_BASE}/${data.video_url}`}
      />

      <div className="images">
        {data.marked_frames?.map((src, i) => (
          <img
            key={i}
            src={`${API_BASE}${src}`}
            alt={`frame-${i}`}
          />
        ))}
      </div>
    </>
  );
}
