// src/components/Modal.jsx
export default function Modal({ visible, onClose, children }) {
  if (!visible) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        width: "100vw",
        height: "100vh",
        backgroundColor: "rgba(0,0,0,0.6)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        zIndex: 9999,
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: "400px",
          backgroundColor: "#fff",
          padding: 20,
          borderRadius: 12,
          color: "#000",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {children}

        <button
          onClick={onClose}
          style={{
            marginTop: 20,
            padding: "10px 16px",
            borderRadius: 8,
            border: "none",
            background: "#1e3a8a",
            color: "#fff",
            cursor: "pointer",
            fontWeight: "bold",
          }}
        >
          Close
        </button>
      </div>
    </div>
  );
}
