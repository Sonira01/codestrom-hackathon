import React, { useState } from 'react';

const CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']; // Example real class names

const Report = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select an image file.');
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        let errorMsg = 'Prediction failed';
        try {
          const errData = await response.json();
          if (errData.error) errorMsg = errData.error;
        } catch {}
        throw new Error(errorMsg);
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Helper to get max risk
  const getMaxRisk = (confidences) => {
    let max = { name: '', value: 0 };
    for (const [name, value] of Object.entries(confidences)) {
      if (value > max.value) max = { name, value };
    }
    return max;
  };

  return (
    <div style={{
      minHeight: '100vh',
      width: '100vw',
      background: 'linear-gradient(135deg, #6366f1 0%, #60a5fa 100%)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      overflow: 'hidden',
    }}>
      <div style={{
        background: 'rgba(255,255,255,0.95)',
        borderRadius: 24,
        boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
        padding: 40,
        maxWidth: 420,
        width: '100%',
        textAlign: 'center',
      }}>
        <h1 style={{
          fontSize: '2.5rem',
          fontWeight: 700,
          marginBottom: 24,
          background: 'linear-gradient(90deg, #6366f1, #60a5fa)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          animation: 'pulse 2s infinite',
        }}>
          Brain Tumor Risk Analysis
        </h1>
        <form onSubmit={handleSubmit} style={{ marginBottom: 20 }}>
          <input type="file" accept="image/*" onChange={handleFileChange} style={{ marginBottom: 16 }} />
          <br />
          <button type="submit" disabled={loading} style={{
            marginTop: 8,
            padding: '12px 32px',
            borderRadius: 12,
            background: 'linear-gradient(90deg, #6366f1, #60a5fa)',
            color: '#fff',
            fontWeight: 600,
            fontSize: '1.1rem',
            border: 'none',
            cursor: loading ? 'not-allowed' : 'pointer',
            boxShadow: '0 2px 8px rgba(99,102,241,0.15)',
            transition: 'background 0.3s',
          }}>
            {loading ? 'Analyzing...' : 'Analyze'}
          </button>
        </form>
        {error && <div style={{ color: '#dc2626', marginTop: 10 }}>{error}</div>}
        {result && result.all_confidences && (
          <div style={{ marginTop: 20, textAlign: 'left' }}>
            <b>Risk per Category:</b>
            {Object.entries(result.all_confidences).map(([name, value]) => (
              <div key={name} style={{ margin: '10px 0' }}>
                <span style={{ display: 'inline-block', width: 120 }}>{name}</span>
                <div style={{ display: 'inline-block', width: 180, verticalAlign: 'middle' }}>
                  <div style={{
                    height: 16,
                    background: '#e0e7ff',
                    borderRadius: 8,
                    overflow: 'hidden',
                  }}>
                    <div style={{
                      width: `${(value * 100).toFixed(1)}%`,
                      height: '100%',
                      background: 'linear-gradient(90deg, #6366f1, #60a5fa)',
                      borderRadius: 8,
                      transition: 'width 0.5s',
                    }}></div>
                  </div>
                </div>
                <span style={{ marginLeft: 10, fontWeight: 600 }}>{(value * 100).toFixed(1)}%</span>
              </div>
            ))}
            <div style={{ marginTop: 24, fontSize: '1.2rem', fontWeight: 700, color: '#2563eb' }}>
              Highest Risk: {getMaxRisk(result.all_confidences).name} ({(getMaxRisk(result.all_confidences).value * 100).toFixed(1)}%)
            </div>
          </div>
        )}
      </div>
      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.7; }
          100% { opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default Report;
