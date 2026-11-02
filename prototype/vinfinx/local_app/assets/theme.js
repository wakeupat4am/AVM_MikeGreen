'use strict';
// Appearance is the only preference stored across sign-outs. No customer data here.
const appearance = (() => {
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  let choice = 'system';
  try { choice = localStorage.getItem('vinfinx.appearance') || 'system'; } catch (_) {}
  if (!['light', 'dark', 'system'].includes(choice)) choice = 'system';
  function apply() {
    document.documentElement.dataset.theme = choice === 'system' ? (media.matches ? 'dark' : 'light') : choice;
    document.querySelectorAll('[data-theme-choice]').forEach(button => {
      button.setAttribute('aria-pressed', String(button.dataset.themeChoice === choice));
    });
  }
  media.addEventListener('change', apply);
  document.addEventListener('DOMContentLoaded', apply);
  document.addEventListener('click', event => {
    const button = event.target.closest('[data-theme-choice]');
    if (!button) return;
    choice = button.dataset.themeChoice;
    try { localStorage.setItem('vinfinx.appearance', choice); } catch (_) {}
    apply();
  });
  apply();
  return { apply };
})();
