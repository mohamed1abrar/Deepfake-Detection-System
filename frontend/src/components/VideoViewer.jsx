const API = import.meta.env.VITE_API_BASE_URL;

export default function VideoViewer({ data }) {
  return (
    <>
      <video controls width="600" src={`${API}${data.video_url}`} />
      <div className="frames">
        {data.marked_frames.map((src) => (
          <img key={src} src={`${API}${src}`} />
        ))}
      </div>
    </>
  );
}
