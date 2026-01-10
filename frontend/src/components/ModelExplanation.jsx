export default function ModelExplanation({ isImage }) {
  return (
    <div className="card explanation">
      <h2>How Detection Works</h2>

      <ul>
        <li>
          <strong>Xception CNN (Video):</strong> Extracts spatial facial artifacts
          across video frames using depthwise separable convolutions.
        </li>

        <li>
          <strong>HF SigLIP Model (Image):</strong> Detects semantic inconsistencies
          in facial texture and lighting.
        </li>

        <li>
          <strong>Ensemble Logic:</strong> Final decision is computed by averaging
          predictions from all available models.
        </li>
      </ul>

      <p className="note">
        {isImage
          ? "For images, only the HF image model is used."
          : "For videos, multiple Xception models analyze facial regions per frame."}
      </p>
    </div>
  );
}
