export default function FrameTable({ frames, onSeek }) {
  return (
    <div className="card">
      <h2>Frame-Level Analysis</h2>

      <div className="frame-table">
        <table>
          <thead>
            <tr>
              <th>Frame</th>
              <th>Confidence</th>
              <th>Label</th>
            </tr>
          </thead>
          <tbody>
            {frames.map((f, i) => (
              <tr key={i} onClick={() => onSeek(f.frame)}>
                <td>{f.frame}</td>
                <td>{(f.avg * 100).toFixed(1)}%</td>
                <td className={f.label}>{f.label}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
