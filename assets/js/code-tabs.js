// The tabs of a post: each <div class="code-tabs"> holds tabs, <div class="tab" data-title="..."> in Markdown, of which
// one is shown at a time, under a row of buttons with their titles. A click on a title shows the tab of that title in
// every group of the page that has one, and the browser keeps the titles picked last for the next pages, where each
// group shows the one of them that it has. Without this script, the tabs are shown one below the other, each under its
// title (_sass/site/_posts.scss).
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

  // the titles picked before, the most recent first
  let saved = [];
  try { saved = JSON.parse(localStorage.getItem('code-tabs')) || []; } catch (e) { /* storage blocked: first tabs */ }
  if (!Array.isArray(saved)) saved = [];

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
        saved = [tab.dataset.title].concat(saved.filter((t) => t !== tab.dataset.title)).slice(0, 20);
        try { localStorage.setItem('code-tabs', JSON.stringify(saved)); } catch (e) { /* the choice is not kept */ }
      });
      bar.appendChild(button);
    });
    group.insertBefore(bar, group.firstChild);
    group.classList.add('code-tabs--ready');
    const titles = tabs.map((t) => t.dataset.title);
    select(group, saved.find((t) => titles.includes(t)) || titles[0]);
  });
})();
