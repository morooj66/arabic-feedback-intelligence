import { Navigate, Route, Routes } from "react-router-dom";
import { LoginPage } from "./pages/LoginPage";
import { RequestAccountPage } from "./pages/RequestAccountPage";
import { ForgotPasswordPage } from "./pages/ForgotPasswordPage";
import { ResetPasswordPage } from "./pages/ResetPasswordPage";
import { ReviewerWorkspace } from "./pages/ReviewerWorkspace";
import { DemoBanner } from "./components/DemoBanner";

export default function App() {
  return (
    <>
    <DemoBanner />
    <Routes>
      <Route path="/" element={<LoginPage />} />
      <Route path="/request-account" element={<RequestAccountPage />} />
      <Route path="/forgot-password" element={<ForgotPasswordPage />} />
      <Route path="/reset-password" element={<ResetPasswordPage />} />
      <Route path="/review" element={<ReviewerWorkspace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
    </>
  );
}
