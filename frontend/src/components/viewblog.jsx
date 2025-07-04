import React from "react";
import { motion } from "framer-motion";

const Blogs = () => {
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

      <div className="relative z-10 px-4 py-12 sm:px-6 lg:px-8 text-white">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            className="text-center mb-16"
          >
            <h1 className="text-4xl font-bold text-blue-300 mb-4">HealthPredict Blog</h1>
            <p className="max-w-2xl mx-auto text-blue-100">
              Dive into the latest in health tech, AI in medicine, and wellness tips from our team.
            </p>
          </motion.div>

          {/* Blog Cards */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2 }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8"
          >
            {[
              {
                title: "The Rise of AI in Healthcare",
                desc: "Discover how artificial intelligence is revolutionizing medical diagnostics.",
                img: "https://source.unsplash.com/600x400/?healthcare,ai",
              },
              {
                title: "Top 5 Health Trends in 2025",
                desc: "From wearable tech to AI-driven diets, here’s what’s trending.",
                img: "https://source.unsplash.com/600x400/?health,technology",
              },
              {
                title: "Preventive Care: Why It Matters",
                desc: "Learn why prevention is better than cure in modern medicine.",
                img: "https://source.unsplash.com/600x400/?doctor,prevention",
              },
            ].map((blog, idx) => (
              <motion.div
                key={idx}
                whileHover={{ scale: 1.03 }}
                className="bg-[#1e3a5f]/70 rounded-xl overflow-hidden shadow-lg"
              >
                <img src={blog.img} alt={blog.title} className="w-full h-48 object-cover" />
                <div className="p-6">
                  <h3 className="text-xl font-semibold text-blue-200 mb-2">{blog.title}</h3>
                  <p className="text-blue-100">{blog.desc}</p>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Blogs;