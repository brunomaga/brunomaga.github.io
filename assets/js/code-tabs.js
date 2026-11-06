// The tabs of a post: each <div class="code-tabs"> holds tabs, <div class="tab" data-title="..."> in Markdown, of which
// one is shown at a time, under a row of buttons with their titles. A click on a title shows the tab of that title in
// every group of the page, and the browser keeps the choice for the next pages. Without this script, the tabs are
// shown one below the other, each under its title (_sass/site/_posts.scss).
(function () {
  const groups = Array.from(document.querySelectorAll('.code-tabs'));
  if (groups.length === 0) return;
  const tabsOf = (group) => Array.from(group.children).filter((c) => c.classList.contains('tab'));

  // shows the tab of the given title in a group, and hides the others; a group without that title stays as it is
  function select(group, title) {
    const tabs = tabsOf(group);
    if (!tabs.some((t) => t.dataset.title === title)) return;
    tabs.forEach((t) => t.classList.toggle('tab--hidden', t.dataset.title !== title));
    group.querySelectorAll(':scope > .tab-bar > button').forEach((b) => {
      b.setAttribute('aria-selected', String(b.textContent === title));
    });
  }

  let saved = null;
  try { saved = localStorage.getItem('code-tab'); } catch (e) { /* storage blocked: start with the first tabs */ }

  groups.forEach((group) => {
    const tabs = tabsOf(group);
    const bar = document.createElement('div');
    bar.className = 'tab-bar';
    bar.setAttribute('role', 'tablist');
    tabs.forEach((tab) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.setAttribute('role', 'tab');
      button.textContent = tab.dataset.title;
      button.addEventListener('click', () => {
        // the groups above the clicked one change height: the page scrolls by as much, so the button stays in place
        const top = button.getBoundingClientRect().top;
        groups.forEach((g) => select(g, tab.dataset.title));
        window.scrollBy(0, button.getBoundingClientRect().top - top);
        try { localStorage.setItem('code-tab', tab.dataset.title); } catch (e) { /* the choice is not kept */ }
      });
      bar.appendChild(button);
    });
    group.insertBefore(bar, group.firstChild);
    group.classList.add('code-tabs--ready');
    select(group, tabs[0].dataset.title);
    if (saved) select(group, saved);
  });
})();
