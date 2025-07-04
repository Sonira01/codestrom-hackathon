import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth, GoogleAuthProvider } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyBDWukfmzrhuUFqEnMM7YmpGEdPkHgC08c",
  authDomain: "health-e867c.firebaseapp.com",
  projectId: "health-e867c",
  storageBucket: "health-e867c.appspot.app", // ⚠️ fixed `.app` to `.com`
  messagingSenderId: "852464890410",
  appId: "1:852464890410:web:f2ed92cfc0ee36439d4953",
  measurementId: "G-LJ0PWTYWPG"
};

const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

export const auth = getAuth();
export const googleProvider = new GoogleAuthProvider();
export default app;