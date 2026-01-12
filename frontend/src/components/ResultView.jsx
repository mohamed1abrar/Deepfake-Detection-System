import VideoResult from "./VideoResult";

const API = import.meta.env.VITE_API_BASE_URL;

export default function ResultView({ data }) {
  return (
    <div className="card">
      <h2>Final Result</h2>

      <p><b>Prediction:</b> {data.ensemble.label}</p>
      <p><b>Confidence:</b> {data.ensemble.avg_fake}</p>

      {data.per_model && (
        <>
          <h3>Model Outputs</h3>
          {data.per_model.map((m) => (
            <p key={m.name}>
              {m.name}: {m.fake_prob} ({m.label})
            </p>
          ))}
        </>
      )}

      {data.is_image ? (
        <div className="images">
          {data.marked_frames.map((src, i) => (
            <img key={i} src={`${API}${src}`} />
          ))}
        </div>
      ) : (
        <VideoResult data={data} />
      )}
    </div>
  );
}
