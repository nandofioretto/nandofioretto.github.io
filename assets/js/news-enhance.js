(function () {
  'use strict';

  // Category detection rules — order matters (first match wins)
  var RULES = [
    // Publications: accepted/appeared at a venue, new preprint, book published
    { cat: 'publication', re: /\baccepted\b.*?\b(at|in|to)\b|new preprint|is now online|will appear|appeared in/i },
    // Funding: grants, research credits, industry programs
    { cat: 'funding',     re: /\bgrant\b|\bfunded\b|\bfunding\b|research credits|google cloud|nvidia.*(?:grant|program)|4-va|cavaliers.*fund|fellowship.*(?:sponsor|will fund)|3cavaliers/i },
    // Awards: paper awards, spotlights, orals, recognitions, fellowships
    { cat: 'award',       re: /\baward\b|best paper|spotlight\b|oral\b|honorable mention|excellence.*research|outstanding.*faculty|recogniti|grfp/i },
    // Talks: given or upcoming
    { cat: 'talk',        re: /gave a talk|keynote|invited talk|summer school|i.ll be teaching|talk at\b|will teach/i },
  ];

  var LABELS = {
    publication: 'Paper',
    award:       'Award',
    talk:        'Talk',
    funding:     'Funding',
    news:        'News'
  };

  function categorize(html) {
    // Strip tags for matching so we don't match inside href="..."
    var text = html.replace(/<[^>]+>/g, ' ');
    for (var i = 0; i < RULES.length; i++) {
      if (RULES[i].re.test(text)) return RULES[i].cat;
    }
    return 'news';
  }

  function renderItem(date, html, cat) {
    return '<li class="news-entry" data-category="' + cat + '">' +
      '<span class="news-entry__date">' + date + '</span>' +
      '<div class="news-entry__body"><div class="news-entry__text">' + html + '</div></div>' +
      '<span class="news-entry__tag">' + LABELS[cat] + '</span>' +
      '</li>';
  }

  function enhance() {
    var stream = document.querySelector('.news-stream');
    if (!stream) return;

    var entries = Array.prototype.slice.call(stream.querySelectorAll('.news-entry'));
    var items = [];

    entries.forEach(function (entry) {
      var dateEl = entry.querySelector('.news-entry__date');
      var textEl = entry.querySelector('.news-entry__text');
      if (!dateEl || !textEl) return;

      var date = dateEl.textContent.trim();
      var raw  = textEl.innerHTML;

      // Split only on <br> that is immediately followed by a bullet marker <strong>-</strong>
      // This preserves <br> used inside multi-line items (e.g. book descriptions)
      var parts = raw.split(/<br\s*\/?>\s*(?=<strong>-<\/strong>)/i);

      parts.forEach(function (part) {
        // Strip the leading bullet marker
        var html = part.replace(/^<strong>-<\/strong>\s*/i, '').trim();
        if (!html) return;
        var cat = categorize(html);
        items.push({ date: date, html: html, cat: cat });
      });
    });

    if (!items.length) return;

    stream.innerHTML = items.map(function (item) {
      return renderItem(item.date, item.html, item.cat);
    }).join('');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enhance);
  } else {
    enhance();
  }
})();
