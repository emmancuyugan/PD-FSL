/**
 * Senyasalin In-App Tutorial — Modal Walkthrough
 *
 * Shows page-specific tutorial modals with Next / Back / Skip controls.
 * Respects a per-user "Don't show again" preference stored in localStorage.
 *
 * Usage:  included automatically via base.html.
 *         Re-trigger from the navbar "?" button or by calling Tutorial.start().
 */

const Tutorial = (() => {
  const TUTORIAL_VERSION = "2026-04-10-v2";

  // ── Step definitions keyed by Flask endpoint name ──────────────────
  const STEPS = {
    tutor: [
      {
        title: "Learn Mode",
        body:  "This page is your <b>study area</b>. Pick a category and sign to watch the demonstration video before practicing.",
      },
      {
        title: "Category & Sign Selection",
        body:  "Signs are grouped into <b>Numbers, Colors, Family, Relationship, and Survival</b>. " +
               "Click a sign badge to load its demo clip.",
      },
      {
        title: "Practice Prep",
        body:  "Focus on hand shape, hand height, and movement path. " +
               "Then move to <b>Select</b> for a specific target sign or <b>Activity</b> for random challenges.",
      },
    ],
    select: [
      {
        title: "Select & Practice Mode",
        body:  "Choose a specific sign, then click <b>Start Practice</b>. " +
               "The flow is: Get Ready (~1s) → Perform (~5s standard mode) → Evaluate.",
      },
      {
        title: "Distance & Framing",
        body:  "Keep one person in frame with face, shoulders, and both hands visible. " +
               "Use the distance guide: <b>green 0.36-0.38 m</b>, yellow near/far bands around it.",
      },
      {
        title: "Feedback Tools",
        body:  "After an incorrect attempt, you can use <b>Show again</b> and <b>Try again</b>. " +
               "The system may display correction guidance and skeleton/video feedback.",
      },
      {
        title: "Mode Options",
        body:  "Use toggles such as <b>Real-Time Recognition</b>, Show Timer, Show Results, and Show Visuals to adjust practice behavior.",
      },
    ],
    activity: [
      {
        title: "Activity Mode",
        body:  "This is challenge practice. The app picks a random sign and you perform it using the same capture/evaluation flow as Select Mode.",
      },
      {
        title: "Challenge Flow",
        body:  "Tap <b>Start Activity</b>, read <b>Your Challenge</b>, perform the sign, and review the result. " +
               "Correct/Incorrect attempts are saved to your results.",
      },
      {
        title: "Retries & Guidance",
        body:  "Use <b>Show again</b> and <b>Try again</b> after feedback to improve the same sign. " +
               "This build uses random sign selection (no visible difficulty selector).",
      },
    ],
    auto_recognition: [
      {
        title: "Auto Recognition Mode",
        body:  "Auto mode continuously evaluates signs and appends recognized results to history while the camera is running.",
      },
      {
        title: "Start / Stop Flow",
        body:  "Click <b>Start Learning!</b> to begin and <b>Stop</b> to end. " +
               "Use Show Results and Real-Time Recognition toggles to adjust live behavior.",
      },
      {
        title: "Recognition Thresholds",
        body:  "Auto mode treats predictions as: <b>&lt; 0.70 Unrecognized</b>, <b>0.70 to &lt; 0.92 Incorrect</b>, and <b>&ge; 0.92 Correct</b>.",
      },
    ],
    results: [
      {
        title: "Your Results",
        body:  "Review your latest attempts, daily activity, and progress summary for the current logged-in account.",
      },
      {
        title: "What You See",
        body:  "Key metrics include <b>Total Attempts</b>, Correct, Incorrect, streak days, and a today summary badge.",
      },
      {
        title: "Export Options",
        body:  "Export a <b>PDF report</b> as either <b>Today only</b> or <b>Full history</b>.",
      },
    ],
  };

  // ── State ──────────────────────────────────────────────────────────
  let currentPage  = "";
  let currentSteps = [];
  let stepIndex    = 0;
  let overlay      = null;
  let modal        = null;
  let userId       = "";

  // ── LocalStorage helpers ───────────────────────────────────────────
  function lsKey() {
    return "senyasalin_tutorial_dismissed_" + TUTORIAL_VERSION + (userId ? "_" + userId : "");
  }

  function isDismissed(page) {
    try {
      const data = JSON.parse(localStorage.getItem(lsKey()) || "{}");
      return !!data[page];
    } catch { return false; }
  }

  function setDismissed(page) {
    try {
      const data = JSON.parse(localStorage.getItem(lsKey()) || "{}");
      data[page] = true;
      localStorage.setItem(lsKey(), JSON.stringify(data));
    } catch { /* ignore */ }
  }

  function resetAll() {
    localStorage.removeItem(lsKey());
  }

  // ── DOM construction ───────────────────────────────────────────────
  function buildOverlay() {
    if (overlay) return;

    overlay = document.createElement("div");
    overlay.id = "tutorial-overlay";

    modal = document.createElement("div");
    modal.id = "tutorial-modal";

    modal.innerHTML = `
      <div class="tut-header">
        <span class="tut-step-label"></span>
        <button class="tut-close" title="Close tutorial">&times;</button>
      </div>
      <h2 class="tut-title"></h2>
      <div class="tut-body"></div>
      <div class="tut-footer">
        <label class="tut-dismiss-label">
          <input type="checkbox" class="tut-dismiss-check" />
          Don't show again on this page
        </label>
        <div class="tut-nav">
          <button class="tut-btn tut-back">← Back</button>
          <button class="tut-btn tut-primary tut-next">Next →</button>
        </div>
      </div>
    `;

    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    // Events
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    modal.querySelector(".tut-close").addEventListener("click", close);
    modal.querySelector(".tut-back").addEventListener("click", back);
    modal.querySelector(".tut-next").addEventListener("click", next);
  }

  // ── Rendering ──────────────────────────────────────────────────────
  function render() {
    if (!modal || !currentSteps.length) return;

    const step = currentSteps[stepIndex];
    modal.querySelector(".tut-step-label").textContent =
      `Step ${stepIndex + 1} of ${currentSteps.length}`;
    modal.querySelector(".tut-title").textContent = step.title;
    modal.querySelector(".tut-body").innerHTML = step.body;

    const backBtn = modal.querySelector(".tut-back");
    const nextBtn = modal.querySelector(".tut-next");

    backBtn.style.visibility = stepIndex === 0 ? "hidden" : "visible";

    if (stepIndex === currentSteps.length - 1) {
      nextBtn.textContent = "Finish";
    } else {
      nextBtn.textContent = "Next →";
    }

    // Entrance animation
    modal.classList.remove("tut-animate");
    void modal.offsetWidth; // reflow
    modal.classList.add("tut-animate");
  }

  // ── Navigation ─────────────────────────────────────────────────────
  function next() {
    if (stepIndex < currentSteps.length - 1) {
      stepIndex++;
      render();
    } else {
      close();
    }
  }

  function back() {
    if (stepIndex > 0) {
      stepIndex--;
      render();
    }
  }

  function close() {
    if (modal) {
      const check = modal.querySelector(".tut-dismiss-check");
      if (check && check.checked) {
        setDismissed(currentPage);
      }
    }
    if (overlay) {
      overlay.classList.add("tut-fade-out");
      setTimeout(() => {
        overlay.classList.remove("tut-fade-out");
        overlay.style.display = "none";
      }, 250);
    }
  }

  // ── Public API ─────────────────────────────────────────────────────

  /** Start the tutorial for a given page. Pass `force=true` to ignore "Don't show again". */
  function start(page, force) {
    page = page || currentPage;
    if (!STEPS[page]) return;
    if (!force && isDismissed(page)) return;

    currentPage  = page;
    currentSteps = STEPS[page];
    stepIndex    = 0;

    buildOverlay();
    overlay.style.display = "flex";
    modal.querySelector(".tut-dismiss-check").checked = false;
    render();
  }

  /** Initialise — call from DOMContentLoaded. */
  function init(pageName, uid) {
    currentPage = pageName;
    userId = uid || "";

    // Show tutorial at least once per user per page, even if previous browser state exists.
    const firstRunKey = "senyasalin_tutorial_first_run_" + TUTORIAL_VERSION + (userId ? "_" + userId : "") + (pageName ? "_" + pageName : "");
    let forceFirstRun = false;
    try {
      forceFirstRun = localStorage.getItem(firstRunKey) !== "1";
    } catch {
      forceFirstRun = false;
    }

    // Auto-show after a brief delay so the page finishes rendering.
    setTimeout(() => {
      start(pageName, forceFirstRun);
      if (forceFirstRun) {
        try { localStorage.setItem(firstRunKey, "1"); } catch { /* ignore */ }
      }
    }, 600);
  }

  return { init, start, resetAll };
})();
