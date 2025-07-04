// import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'

import './App.css'
import Home from './components/Home'
import About from './components/About'
import Blog from './components/Blog'
import Report from './components/Report'
import Navbar from './components/Navbar'
import Login from './components/Login'
import Register from './components/Register'

function App() {

  return (
  <>
  <BrowserRouter>
  <Navbar />
  <Routes>
    <Route path='/' element={<Home />} />
    <Route path='/signin' element={<Login />} />
    <Route path='/about' element={<About />} />
    <Route path='/signup' element={<Register />} />
    <Route path='/blog' element={<Blog />} />
    <Route path='*' element={<h1>404 Not Found</h1>} />
    <Route path='/predict' element={< Report/>} />
    </Routes>
    </BrowserRouter>
  </>
  )
}

export default App
