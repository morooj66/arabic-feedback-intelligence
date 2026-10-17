export function BrandLogo({ className = "brand-logo" }: { className?: string }) {
  return (
    <img
      className={className}
      src="/brand-logo.png"
      alt="شعار المشروع"
    />
  );
}
