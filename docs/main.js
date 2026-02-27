const themes = ["dark", "cool", "aurora"];

function resolveInitialTheme() {
  const savedTheme = window.localStorage.getItem("rf-docs-theme");
  if (savedTheme && themes.includes(savedTheme)) {
    return savedTheme;
  }

  return document.documentElement.getAttribute("data-theme") || "dark";
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const toggle = document.getElementById("theme-toggle");
  const index = themes.indexOf(theme);
  const next = themes[(index + 1) % themes.length];
  if (toggle) {
    toggle.textContent = `Switch to ${next}`;
  }
}

function bindTheme() {
  const toggle = document.getElementById("theme-toggle");
  if (!toggle) return;

  let current = resolveInitialTheme();
  setTheme(current);

  toggle.addEventListener("click", () => {
    const index = themes.indexOf(document.documentElement.getAttribute("data-theme"));
    const next = themes[(index + 1) % themes.length];
    current = next;
    setTheme(current);
    window.localStorage.setItem("rf-docs-theme", current);
  });
}

function markActiveNav() {
  const path = window.location.pathname.split("/").pop() || "index.html";
  const links = document.querySelectorAll(".topbar nav a");
  links.forEach((link) => {
    const target = link.getAttribute("href");
    if (target === path) {
      link.classList.add("active");
      link.setAttribute("aria-current", "page");
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  bindTheme();
  markActiveNav();
});
