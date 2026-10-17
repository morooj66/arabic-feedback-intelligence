import type { ReactNode } from "react";
import "../pages/LoginPage.css";

/** The login screen's page chrome (photo column, botanical ornament, brand logo, copyright), extracted unchanged so the password-reset screens
 * share exactly the same visual identity. `children` is the card (a <form> or <div className="login-card">). */
export function AuthShell({ children }: { children: ReactNode }) {
  return (
    <main className="login-page screen-login" dir="rtl">
      <div className="login-photo-col">
        <img className="login-building" src="/login-photo.png" alt="" />
        <div className="login-photo-shade" aria-hidden="true" />
        <div className="login-photo-copy">
          <h1>نظام مراجعة تعليقات الزوار</h1>
          <span className="login-photo-dash" aria-hidden="true" />
          <p className="login-photo-tagline">صوت الزائر محل اهتمام</p>
        </div>
      </div>

      <section className="login-panel">
        <svg className="login-botanical" viewBox="0 0 420 700" fill="none" aria-hidden="true" focusable="false">
          <path d="M390 700C355 450 215 230 60 30" stroke="currentColor" strokeWidth="3" />
          <g fill="currentColor">
            <path d="M104 91Q40 82 4 12Q76 37 104 91Z" />
            <path d="M139 139Q150 65 115 0Q105 78 139 139Z" />
            <path d="M172 187Q76 172 18 89Q118 115 172 187Z" />
            <path d="M202 235Q229 133 184 49Q170 154 202 235Z" />
            <path d="M235 294Q109 265 37 177Q174 207 235 294Z" />
            <path d="M265 351Q304 240 268 134Q235 266 265 351Z" />
            <path d="M297 417Q159 384 81 287Q235 330 297 417Z" />
            <path d="M324 481Q368 351 345 245Q291 377 324 481Z" />
            <path d="M348 549Q218 518 132 409Q279 457 348 549Z" />
            <path d="M370 621Q416 488 407 379Q346 502 370 621Z" />
          </g>
        </svg>
        <img className="login-panel-logo" src="/login-logo.png" alt="شعار المشروع" />
        {children}
        <p className="login-copyright">© 2026 RCS</p>
      </section>
    </main>
  );
}
