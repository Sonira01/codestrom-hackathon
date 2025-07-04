import React, { useState, useEffect } from 'react';
import Dna4 from '../assets/videos/Dna4.mp4';
import { Link } from 'react-router-dom';
import { motion } from "framer-motion";
import { useNavigate } from 'react-router-dom';
import Footer from './Footer';
import { auth } from '../firebase';

const Home = () => {
  const phrases = [
    "Always for your Health Care",
    "AI model powered report result",
    "Get AI result",
  ];
  const navigate = useNavigate();
  
  const [displayText, setDisplayText] = useState("");
  const [currentPhraseIndex, setCurrentPhraseIndex] = useState(0);
  const [isDeleting, setIsDeleting] = useState(false);
  const [typingSpeed, setTypingSpeed] = useState(150);

  useEffect(() => {
    const currentPhrase = phrases[currentPhraseIndex];
    
    const handleTyping = () => {
      if (!isDeleting) {
        // Typing forward
        if (displayText.length < currentPhrase.length) {
          setDisplayText(currentPhrase.substring(0, displayText.length + 1));
          setTypingSpeed(150);
        } else {
          // Finished typing, pause then start deleting
          setTimeout(() => {
            setIsDeleting(true);
            setTypingSpeed(50);
          }, 1000);
        }
      } else {
        // Deleting backward
        if (displayText.length > 0) {
          setDisplayText(displayText.substring(0, displayText.length - 1));
          setTypingSpeed(50);
        } else {
          // Finished deleting, move to next phrase
          setIsDeleting(false);
          setCurrentPhraseIndex((prevIndex) => 
            (prevIndex + 1) % phrases.length
          );
          setTypingSpeed(150);
        }
      }
    };

    const timer = setTimeout(handleTyping, typingSpeed);
    return () => clearTimeout(timer);
  }, [displayText, isDeleting, currentPhraseIndex, phrases]);

  return (
    <div className="h-screen overflow-hidden">
      {/* Video Background */}
      <div className="absolute inset-0 w-full h-full">
        <video 
          className="w-full h-full object-cover"
          autoPlay 
          loop 
          muted 
          playsInline
          preload="auto"
        >
          <source src={Dna4} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      </div>
      
      {/* Content Overlay */}
      <div className="ml-[-700px] -mt-[-2px] relative z-10 h-[86vh] flex items-center justify-center left-0 top-0">
        <div className="text-white text-4xl md:text-7xl p-6 rounded-2xl h-80 w-120 absolute ">
          {displayText}
          <span className="ml-1 animate-pulse">|</span>
          <motion.div
      whileTap={{ scale: 0.95 }}
      whileHover={{ scale: 1.05 }}
      onClick={() => {if(auth.user){navigate("/report")}else{navigate("/signin")}}}
      className=" absolute top-70 left-25 p-2 text-2xl bg-gradient-to-r from-indigo-600 to-blue-700 text-white font-semibold px-6 py-3 rounded-xl shadow-lg cursor-pointer select-none transition-all duration-300 ease-in-out hover:from-blue-700 hover:to-indigo-800"
    >
      Get Started
    </motion.div>
          {/* <Link to='/report'><div className="absolute top-70 left-35 p-2 text-2xl">GET STARTED</div>
        </Link> */}
        </div>
          {/* <div className="absolute top-0 right-0 p-2"><button>get started</button></div> */}
      </div>
      
      {/* <footer className="z-10 bg-gray-800 text-white py-8 px-4 mt-auto">
        <div className="container mx-auto text-center text-sm">
          <p>&copy; {new Date().getFullYear()} HealthPredict. All rights reserved.</p>
          <div className="mt-2">
            <a href="#" className="text-gray-400 hover:text-white mx-2">Privacy Policy</a>
            <span className="text-gray-500">|</span>
            <a href="#" className="text-gray-400 hover:text-white mx-2">Terms of Service</a>
          </div>
        </div>
        </footer> */}
        <Footer />
    </div>
  );
};

export default Home;