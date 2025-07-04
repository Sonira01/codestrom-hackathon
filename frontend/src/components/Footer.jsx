import React from "react";
import { motion } from "framer-motion";

const Footer = () => {
  return (
    <motion.footer
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="bg-slate-950/90 text-blue-300 py-7 px-4 mt-auto backdrop-blur-sm border-t border-blue-800"
    >
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
        <div className="text-sm text-center md:text-left">
          &copy; {new Date().getFullYear()} <span className="text-cyan-400 font-semibold">HealthPredict</span>. All rights reserved.
        </div>

        <div className="flex space-x-4 text-sm">
          <a href="#" className="hover:text-cyan-300 transition">Privacy Policy</a>
          <span className="text-blue-700">|</span>
          <a href="#" className="hover:text-cyan-300 transition">Terms of Service</a>
          <span className="text-blue-700">|</span>
          <a href="#" className="hover:text-cyan-300 transition">Contact</a>
        </div>
      </div>
    </motion.footer>
  );
};

export default Footer;
