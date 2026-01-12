import VideoViewer from "./VideoViewer";

const API = import.meta.env.VITE_API_BASE_URL;

export default function ResultPanel({ data }) {
  return (
    <div className="card">
      <h2>Result</h2>

      <p><b>Prediction:</b> {data.ensemble.label}</p>
      <p><b>Confidence:</b> {data.ensemble.avg_fake}</p>

      <h3>Models</h3>
      {data.per_model.map((m) => (
        <p key={m.name}>
          {m.name}: {m.fake_prob} ({m.label})
        </p>
      ))}

      {data.is_image ? (
        data.marked_frames.map((src) => (
          <img key={src} src={`${API}${src}`} />
        ))
      ) : (
        <VideoViewer data={data} />
      )}
    </div>
  );
}
