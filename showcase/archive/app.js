// Tab switcher
document.querySelectorAll('.tabs').forEach((group) => {
  const name = group.dataset.tabgroup;
  group.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.target;
      group.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      document.querySelectorAll(`.tab-panel[data-tabgroup="${name}"]`).forEach((p) => {
        p.classList.toggle('active', p.dataset.target === target);
      });
    });
  });
});

// Image click-to-zoom lightbox
const lb = document.getElementById('lightbox');
if (lb) {
  const lbImg = document.getElementById('lightbox-img');
  const lbCap = document.getElementById('lightbox-cap');
  document.querySelectorAll('.fig img').forEach((img) => {
    img.addEventListener('click', () => {
      lbImg.src = img.src;
      lbImg.alt = img.alt;
      lbCap.textContent = img.dataset.caption || img.alt;
      lb.classList.add('open');
    });
  });
  const close = () => lb.classList.remove('open');
  lb.addEventListener('click', close);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') close();
  });
}
