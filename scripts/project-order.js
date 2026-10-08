// Projects page: order the project groups for the visitor role picked in the
// chat. The role lives in the chat widget's tab-scoped session store
// (chat-widget.js, STORE_KEY), so a role picked on any page carries over
// here; picking one on this page reorders the groups at once
// ('ycchat-role'). The markup's own order is the default, so without a role,
// storage or this script the page reads game -> agentic -> other.
(function () {
  'use strict';

  var STORE_KEY = 'yc-chat-session';
  var DEFAULT = ['game', 'agentic', 'other'];
  var ORDERS = {
    ai_agent_recruiter: ['agentic', 'game', 'other'],
    combat_design_recruiter: ['game', 'other', 'agentic']
  };

  var groups = {};
  var list = document.querySelectorAll('.project-group[data-group]');
  if (!list.length) return;
  for (var i = 0; i < list.length; i++) groups[list[i].getAttribute('data-group')] = list[i];
  var parent = list[0].parentNode;
  var anchor = list[list.length - 1].nextSibling;

  function storedRole() {
    try {
      var data = JSON.parse(sessionStorage.getItem(STORE_KEY) || 'null');
      return data && data.role || null;
    } catch (e) { return null; }
  }

  // Moves nodes rather than restyling them, so reading and tab order follow
  // what is on screen; the card numbers (a CSS counter) follow too.
  function apply(role) {
    var order = ORDERS[role] || DEFAULT;
    for (var j = 0; j < order.length; j++) {
      if (groups[order[j]]) parent.insertBefore(groups[order[j]], anchor);
    }
  }

  apply(storedRole());
  window.addEventListener('ycchat-role', function (e) { apply(e.detail && e.detail.role); });
})();
