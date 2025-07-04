import React, { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { motion } from "framer-motion";
import { auth } from "../firebase/config";
import { onAuthStateChanged, signOut } from "firebase/auth";
import { toast } from "react-toastify";


const Navbar = () => {
  const navigate = useNavigate();

  // Navigation items
  const navItems = [
    { path: "/", name: "Home", icon: "M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h12a1 1 0 001-1v-10" },
    { path: "/about", name: "About", icon: "M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" },
    { path: "/blog", name: "Blog", icon: "M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" },
    { path: "/report", name: "Report", icon: "M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" }
  ];

  const authItems = [
    { path: "/signin", name: "Sign In", icon: "M11 16l-4-4m0 0l4-4m-4 4h14m-5 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h7a3 3 0 013 3v1" },
    { path: "/signup", name: "Sign Up", icon: "M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" }
  ];

  // Animations
  const logoVariants = {
    hidden: { x: -100, opacity: 0 },
    visible: { x: 0, opacity: 1, transition: { type: "spring", stiffness: 100, damping: 10 } }
  };

  const navItemVariants = {
    hidden: { y: -20, opacity: 0 },
    visible: (i) => ({
      y: 0, opacity: 1,
      transition: { delay: i * 0.1, type: "spring", stiffness: 100 }
    })
  };

   const [user, setUser] = useState(null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
    return () => unsubscribe();
  }, []);

  const handleLogout = () => {
    signOut(auth)
      .then(() => {
        toast.warn("Logged out successfully!");
      })
      .catch((error) => {
        toast.error("Error logging out: " + error.message);
      });
  };

  const authItemVariants = {
    hidden: { x: 100, opacity: 0 },
    visible: (i) => ({
      x: 0, opacity: 1,
      transition: { delay: i * 0.1, type: "spring", stiffness: 100 }
    })
  };

  return (
    <div className="sticky top-0 z-50 backdrop-blur-sm shadow-md bg-slate-950/80">
      <nav className="p-4">
        <div className="container mx-auto flex justify-between items-center">
          {/* Logo */}
          <motion.div initial="hidden" animate="visible" variants={logoVariants}>
            <Link
              to="/"
              className="flex items-center text-2xl font-bold text-blue-400 hover:text-cyan-300 transition duration-300"
            >
              <svg className="w-8 h-8 mr-2 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z"
                  clipRule="evenodd"
                />
              </svg>
              HealthPredict
            </Link>
          </motion.div>

          {/* Navigation */}
          <div className="hidden md:flex items-center space-x-3">
            {navItems.map((item, i) => (
              <motion.button
                key={item.path}
                custom={i}
                initial="hidden"
                animate="visible"
                variants={navItemVariants}
                onClick={() => navigate(item.path)}
                className="flex items-center px-4 py-2 text-blue-300 hover:text-cyan-300 hover:bg-blue-950/10 rounded-lg transition-all duration-300"
              >
                <svg
                  className="w-5 h-5 mr-2 text-blue-400 group-hover:text-cyan-300"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={item.icon} />
                </svg>
                {item.name}
              </motion.button>
            ))}
          </div>
            
          {/* Auth Buttons */}
          {/* <motion.div className="flex items-center space-x-2" initial="hidden" animate="visible">
            {authItems.map((item, i) => (
              <motion.button
                key={item.path}
                custom={i}
                variants={authItemVariants}
                onClick={() => navigate(item.path)}
                className={`flex items-center px-4 py-2 rounded-lg transition duration-300
                  ${
                    item.name === "Sign Up"
                      ? "text-white bg-gradient-to-r from-indigo-600 to-blue-700 hover:from-blue-700 hover:to-indigo-800"
                      : "text-blue-300 hover:text-cyan-300 hover:bg-blue-950/10"
                  }
                `}
              >
                <svg
                  className={`w-5 h-5 mr-2 ${
                    item.name === "Sign Up" ? "text-white" : "text-blue-400"
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={item.icon} />
                </svg>
                <span className="hidden md:inline">{item.name}</span>
              </motion.button>
            ))}
          </motion.div> */}
          {user ? (
  <>
    <span className="text-blue-300 ml - 10">Welcome, {user.displayName || user.email}</span>
    <button
      onClick={handleLogout}
      className="text-white bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg transition duration-300"
    >
      Logout
    </button>
  </>
) : (
  <>
    {/* Auth Buttons */}
    <motion.div className="flex items-center space-x-2" initial="hidden" animate="visible">
      {authItems.map((item, i) => (
        <motion.button
          key={item.path}
          custom={i}
          variants={authItemVariants}
          onClick={() => navigate(item.path)}
          className={`flex items-center px-4 py-2 rounded-lg transition duration-300
            ${
              item.name === "Sign Up"
                ? "text-white bg-gradient-to-r from-indigo-600 to-blue-700 hover:from-blue-700 hover:to-indigo-800"
                : "text-blue-300 hover:text-cyan-300 hover:bg-blue-950/10"
            }
          `}
        >
          <svg
            className={`w-5 h-5 mr-2 ${
              item.name === "Sign Up" ? "text-white" : "text-blue-400"
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={item.icon} />
          </svg>
          <span className="hidden md:inline">{item.name}</span>
        </motion.button>
      ))}
    </motion.div>
  </>
)}

        </div>
      </nav>
    </div>
  );
};

export default Navbar;