import React, { useState } from 'react';

const CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor'];

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
      const response = await fetch(`${import.meta.env.VITE_API_URL}/predict`, {
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

  const getMaxRisk = (confidences) => {
    let max = { name: '', value: 0 };
    for (const [name, value] of Object.entries(confidences)) {
      if (value > max.value) max = { name, value };
    }
    return max;
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Background Video */}
      <video
        className="fixed top-0 left-0 w-full h-full object-cover z-[-1]"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        poster="https://i.pinimg.com/videos/thumbnails/originals/2e/fc/a0/2efca0faa924c69e71e576da04168958.0000000.jpg"
        src="https://cdn.pixabay.com/video/2023/04/02/157741-818722985_large.mp4"
      ></video>

      {/* Overlay */}
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-[#0d1b2a]/80 to-[#1b263b]/90 z-0"></div>

      {/* Content */}
      <div className="relative z-10 flex items-center justify-center min-h-screen">
        <div className="bg-white/90 backdrop-blur-md rounded-3xl shadow-xl px-10 py-12 max-w-md w-full text-center">
          <h1 className="text-3xl font-bold mb-6 bg-gradient-to-r from-[#3B536A] via-[#1E3248] to-[#0C1D2E] bg-clip-text text-transparent">
  Brain Tumor Risk Analysis
</h1>

          <form onSubmit={handleSubmit} className="space-y-6">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-700 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-indigo-100 file:text-indigo-700 hover:file:bg-indigo-200"
            />

            <button
              type="submit"
              disabled={loading}
              className={`w-full py-3 text-lg font-semibold rounded-xl text-white shadow-md transition duration-300 
                ${loading
                  ? 'bg-indigo-300 cursor-not-allowed'
                  : 'bg-gradient-to-r from-indigo-600 to-blue-500 hover:from-indigo-700 hover:to-blue-600'
                }`}
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>

            {/* Progress Bar */}
            {loading && (
              <div className="w-full h-2 bg-indigo-100 rounded overflow-hidden">
                <div className="w-1/3 h-full bg-gradient-to-r from-indigo-500 to-blue-400 animate-pulse"></div>
              </div>
            )}
          </form>

          {/* Error Message */}
          {error && (
            <div className="text-red-600 font-medium mt-4">{error}</div>
          )}

          {/* Prediction Result */}
          {result?.all_confidences && (
            <div className="mt-6 text-left">
              <h2 className="font-semibold mb-2 text-gray-800">Risk per Category:</h2>
              {Object.entries(result.all_confidences).map(([name, value]) => (
                <div key={name} className="mb-3">
                  <div className="flex justify-between text-sm text-gray-700">
                    <span>{name}</span>
                    <span className="font-semibold">{(value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-3 bg-indigo-100 rounded">
                    <div
                      className="h-full bg-gradient-to-r from-indigo-500 to-blue-400 rounded transition-all duration-500"
                      style={{ width: `${(value * 100).toFixed(1)}%` }}
                    ></div>
                  </div>
                </div>
              ))}
              <div className="mt-5 text-blue-600 font-bold text-lg">
                Highest Risk: {getMaxRisk(result.all_confidences).name} ({(getMaxRisk(result.all_confidences).value * 100).toFixed(1)}%)
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Report;
