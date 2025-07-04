import React from "react";
import { motion } from "framer-motion";

const About = () => {
  const handleSubmit = (e) => {
    e.preventDefault();
    alert("Thank you for your message! We'll get back to you soon.");
    e.target.reset();
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
        src="https://cdn.pixabay.com/video/2023/04/02/157741-818722985_large.mp4" // Replace with working .mp4 if blob doesn't play
      ></video>

      {/* Overlay */}
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-[#0d1b2a]/80 to-[#1b263b]/90 z-0"></div>

      <div className="relative z-10 px-4 py-12 sm:px-6 lg:px-8 text-white">
        <div className="max-w-7xl mx-auto">
          {/* About Section */}
          <motion.div
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            className="text-center mb-16"
          >
            <h1 className="text-4xl font-bold text-blue-300 mb-6">
              About HealthPredict
            </h1>
            <div className="max-w-3xl mx-auto text-lg text-blue-100">
              <p className="mb-6">
                HealthPredict is an innovative platform that leverages AI to provide personalized health predictions and recommendations.
              </p>
              <p className="mb-6">
                Founded in 2025 by a team of enjeneering professionals and enthusiastic developer, we aim to make healthcare more preventive and accessible.
              </p>

              {/* Social Links */}
               <div className="flex justify-center space-x-6 mt-8">
              {/* GitHub */}
              <a
                href="https://github.com/healthpredict"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-white transition transform hover:scale-110"
              >
                <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0C5.372 0 0 5.373 0 12a12 12 0 008.207 11.385c.6.111.793-.26.793-.577v-2.234c-3.338.726-4.034-1.416-4.034-1.416-.546-1.387-1.334-1.756-1.334-1.756-1.089-.745.084-.729.084-.729 1.205.084 1.84 1.237 1.84 1.237 1.069 1.834 2.806 1.304 3.492.997.107-.775.418-1.305.762-1.604C6.42 17.688 3.618 16.659 3.618 12.063c0-1.312.468-2.382 1.236-3.222-.123-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.3 1.23a11.51 11.51 0 013.003-.404c1.02.005 2.047.138 3.006.404 2.292-1.552 3.298-1.23 3.298-1.23.652 1.653.241 2.874.117 3.176.77.84 1.236 1.91 1.236 3.222 0 4.609-2.808 5.624-5.479 5.92.43.372.824 1.102.824 2.222v3.294c0 .318.192.692.801.576A12.003 12.003 0 0024 12c0-6.627-5.373-12-12-12z" />
                </svg>
              </a>

              {/* LinkedIn */}
              <a
                href="https://linkedin.com/company/healthpredict"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-white transition transform hover:scale-110"
              >
                <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M4.983 3.5a2.5 2.5 0 11-.002 5.001 2.5 2.5 0 01.002-5zM2 9h6v12H2zm7.5 0H16v1.8c.8-1.3 3-2 4.5-2 4.5 0 5.5 3 5.5 7v5.2h-6V16c0-1.5-.5-2.5-2-2.5-1.5 0-2.5 1-2.5 2.5v5h-6V9z" />
                </svg>
              </a>

              {/* Instagram */}
              <a
                href="https://instagram.com/healthpredict"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-300 hover:text-white transition transform hover:scale-110"
              >
                <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 2.2c3.2 0 3.6.01 4.9.07 3.2.15 4.8 1.7 4.9 4.9.06 1.3.07 1.6.07 4.8s-.01 3.6-.07 4.9c-.15 3.2-1.7 4.8-4.9 4.9-1.3.06-1.6.07-4.9.07s-3.6-.01-4.9-.07c-3.2-.15-4.8-1.7-4.9-4.9C2.2 15.6 2.2 15.3 2.2 12s.01-3.6.07-4.9C2.4 3.9 4 2.4 7.2 2.2 8.5 2.2 8.8 2.2 12 2.2zM12 0C8.7 0 8.3.01 7 .07 2.6.27.3 2.6.07 7 .01 8.3 0 8.7 0 12s.01 3.7.07 5c.2 4.4 2.6 6.7 7 6.9 1.3.06 1.7.07 5 .07s3.7-.01 5-.07c4.4-.2 6.7-2.6 6.9-7 .06-1.3.07-1.7.07-5s-.01-3.7-.07-5C23.7 2.6 21.4.3 17 .07 15.7.01 15.3 0 12 0zm0 5.8a6.2 6.2 0 100 12.4 6.2 6.2 0 000-12.4zm0 10.2a4 4 0 110-8 4 4 0 010 8zm6.4-11.8a1.4 1.4 0 11-2.8 0 1.4 1.4 0 012.8 0z" />
                </svg>
              </a>
            </div>
            </div>
          </motion.div>

          {/* Team Section */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.2 }}
            className="mb-16"
          >
            <h2 className="text-3xl font-bold text-blue-300 text-center mb-12">Our Team</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[
                {
                  name: "Mr. Rakshith Ganjimut",
                  role: "Front-end Developer",
                  bio: "2nd year B. Tech at VIT,Chennai.",
                  img: "https://imgs.search.brave.com/k_0L-TWQJQnLSpwz6eY-cMJ5Mtv1vJvH7gY2fHC56vk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzL2U2L2U0/L2RmL2U2ZTRkZjI2/YmE3NTIxNjFiOWZj/NmExNzMyMWZhMjg2/LmpwZw",
                },
                {
                  name: "Ms. Soumya Gupta",
                  role: "Backend Developer and ML Engineer",
                  bio: "2nd year B. Tech at VIT,Chennai.",
                  img: "https://imgs.search.brave.com/k_0L-TWQJQnLSpwz6eY-cMJ5Mtv1vJvH7gY2fHC56vk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzL2U2L2U0/L2RmL2U2ZTRkZjI2/YmE3NTIxNjFiOWZj/NmExNzMyMWZhMjg2/LmpwZw",
                },
                {
                  name: "Mr. Nirvik Goswami",
                  role: "Full Stack Developer and ML Engineer",
                  bio: "2nd year B. Tech at VIT,Chennai.",
                  img: "https://imgs.search.brave.com/k_0L-TWQJQnLSpwz6eY-cMJ5Mtv1vJvH7gY2fHC56vk/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9pLnBp/bmltZy5jb20vb3Jp/Z2luYWxzL2U2L2U0/L2RmL2U2ZTRkZjI2/YmE3NTIxNjFiOWZj/NmExNzMyMWZhMjg2/LmpwZw",
                },
                // {
                //   name: "Mark Chen",
                //   role: "Lead Data Scientist",
                //   bio: "Machine learning expert specializing in healthcare applications and predictive modeling.",
                //   img: "https://images.unsplash.com/photo-1560250097-0b93528c311a",
                // },
                // {
                //   name: "Emily Rodriguez",
                //   role: "Product Director",
                //   bio: "Healthcare technology veteran focused on creating user-centered health solutions.",
                //   img: "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2",
                // },
              ].map((member, index) => (
                <motion.div
                  key={index}
                  whileHover={{ scale: 1.05 }}
                  className="bg-[#1e3a5f]/70 text-white p-6 rounded-xl shadow-md"
                >
                  <div className="w-32 h-32 mx-auto mb-4 overflow-hidden rounded-full border-4 border-blue-200">
                    <img src={member.img} alt={member.name} className="w-full h-full object-cover" />
                  </div>
                  <h3 className="text-xl font-bold text-blue-200 text-center">{member.name}</h3>
                  <p className="text-blue-400 text-center mb-3">{member.role}</p>
                  <p className="text-blue-100 text-center">{member.bio}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* Contact Section */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.4 }}
            className="bg-[#1b263b]/80 text-white rounded-xl shadow-lg overflow-hidden max-w-4xl mx-auto"
          >
            <div className="md:flex">
              <div className="md:w-1/2 bg-[#1e3a5f] p-8">
                <h2 className="text-2xl font-bold mb-6">Get in Touch</h2>
                <p className="mb-6">
                  Have questions? Reach out and we’ll get back to you soon.
                </p>
                <div className="space-y-4 text-blue-200">
                  <div className="flex items-center">
                    <span>📧 contact@healthpredict.com</span>
                  </div>
                  <div className="flex items-center">
                    <span>📞 +91 123213213</span>
                  </div>
                  <div className="flex items-center">
                    <span>📍 Chennai, India</span>
                  </div>
                </div>
              </div>
              <div className="md:w-1/2 p-8">
                <h2 className="text-2xl font-bold text-blue-300 mb-6">Contact Us</h2>
                <form onSubmit={handleSubmit}>
                  <input
                    type="text"
                    name="name"
                    required
                    placeholder="Your Name"
                    className="w-full mb-4 px-4 py-2 border border-blue-300 bg-[#1b263b]/70 text-white rounded-lg"
                  />
                  <input
                    type="email"
                    name="email"
                    required
                    placeholder="Your Email"
                    className="w-full mb-4 px-4 py-2 border border-blue-300 bg-[#1b263b]/70 text-white rounded-lg"
                  />
                  <textarea
                    name="comment"
                    rows="4"
                    required
                    placeholder="Your Message"
                    className="w-full mb-6 px-4 py-2 border border-blue-300 bg-[#1b263b]/70 text-white rounded-lg"
                  ></textarea>
                  <button
                    type="submit"
                    className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg font-medium transition transform hover:scale-105"
                  >
                    Send Message
                  </button>
                </form>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default About;
