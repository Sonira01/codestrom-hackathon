import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { signInWithEmailAndPassword, signInWithPopup, GoogleAuthProvider  } from "firebase/auth";
import { auth } from "../firebase"; // Ensure firebase config is set
import { toast } from "react-toastify";

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();
  const googleProvider = new GoogleAuthProvider();

  const handleLogin = async (e) => {
    e.preventDefault();
    try {
      await signInWithEmailAndPassword(auth, email, password);
      toast.success("Login Successfull")
      navigate("/");
    } catch (err) {
      toast.err("Login failed. Please check your credentials.");
    }
  };
  const handleGoogleSignUp = async () => {
      try {
        await signInWithPopup(auth, googleProvider);
      toast.success("Google Sign-Up successful!");
        navigate("/report");
      } catch (error) {
        toast.error("Google Sign-Up failed: " + error.message);
        console.log(error.message)
      }
    };

  return (
    <div className="relative h-[87vh] overflow-hidden">
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
      <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-[#0d1b2a]/80 to-[#1b263b]/90 z-0"></div>

      <div className="relative z-10 flex flex-col justify-center items-center h-screen">
        <form
          onSubmit={handleLogin}
          className="bg-[#1e3a5f]/80 p-8 rounded-xl shadow-lg max-w-md w-full text-white"
        >
          <h2 className="text-3xl font-bold text-blue-300 mb-6 text-center">Login</h2>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            required
            className="w-full mb-4 px-4 py-2 border border-blue-300 bg-[#1b263b]/70 text-white rounded-lg"
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            required
            className="w-full mb-6 px-4 py-2 border border-blue-300 bg-[#1b263b]/70 text-white rounded-lg"
          />
          <button
            type="submit"
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg font-medium transition transform hover:scale-105"
          >
            Login
          </button>
        </form>
        <button
           onClick={handleGoogleSignUp}
            className="w-[70vh] mt-2.5 bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg font-medium transition transform hover:scale-105"
          >
            SignIn with Google
          </button>
      </div>
    </div>
  );
};

export default Login;