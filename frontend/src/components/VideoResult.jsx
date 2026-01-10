export default function VideoResult({ data }) {
  return (
    <>
      <video
        controls
        width="600"
        src={`http://127.0.0.1:8000/${data.video_url}`}
      />

      <div className="images">
        {data.marked_frames?.map((src, i) => (
          <img key={i} src={`http://127.0.0.1:8000${src}`} />
        ))}
      </div>
    </>
  );
}
