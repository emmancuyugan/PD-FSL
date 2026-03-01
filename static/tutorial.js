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
  // ── Step definitions keyed by Flask endpoint name ──────────────────
  const STEPS = {
    tutor: [
      {
        title: "Learn Mode",
        body:  "This page plays <b>demonstration videos</b> for each Filipino Sign Language sign. " +
               "Watch closely and try to imitate the hand movements shown.",
      },
      {
        title: "Category & Sign Selection",
        body:  "Browse signs organized by category (Colors, Family, Numbers, Relationships, Survival). " +
               "Select a category, then pick a specific sign to see its video.",
      },
      {
        title: "Playback Controls",
        body:  "Use the video controls to <b>pause, replay, and slow down</b> the demonstration so you can study each gesture at your own pace.",
      },
    ],
    select: [
      {
        title: "Select & Practice Mode",
        body:  "Choose a specific sign you want to practice. The system will ask you to perform it in front of your camera.",
      },
      {
        title: "Camera Feed",
        body:  "Your webcam feed appears on screen. Position yourself so your <b>hands and upper body</b> are clearly visible.",
      },
      {
        title: "Ghost Overlay",
        body:  "A translucent <b>ghost overlay</b> shows the correct pose landmarks. Try to match your movements to the ghost for best results.",
      },
      {
        title: "Recognition Feedback",
        body:  "After performing the sign, the AI will evaluate your gesture and show a confidence score. " +
               "Aim for high confidence to master each sign!",
      },
    ],
    activity: [
      {
        title: "Activity Mode",
        body:  "Test your skills! The system presents a sign for you to perform, then evaluates your attempt.",
      },
      {
        title: "Timer & Scoring",
        body:  "You'll have a limited time to perform the sign. Results are saved automatically and contribute to your progress tracking.",
      },
      {
        title: "Difficulty",
        body:  "Choose your difficulty level to adjust the complexity of the signs presented and the strictness of evaluation.",
      },
    ],
    auto_recognition: [
      {
        title: "Auto Recognition Mode",
        body:  "Continuous, real-time sign language recognition. Just sign naturally and the AI identifies each gesture as you go.",
      },
      {
        title: "How It Works",
        body:  "The AI collects a sequence of your pose landmarks and runs inference continuously. " +
               "Results appear on screen in real time.",
      },
      {
        title: "Best Practices",
        body:  "Ensure good lighting, keep your hands within the camera frame, and perform signs at a natural pace for the most accurate recognition.",
      },
    ],
    results: [
      {
        title: "Your Results",
        body:  "View your complete practice history — every sign you've practiced, with confidence scores and timestamps.",
      },
      {
        title: "Summary Stats",
        body:  "See your <b>daily count</b>, <b>7-day total</b>, <b>current streak</b>, and <b>most-practiced sign</b> at a glance.",
      },
      {
        title: "Export Options",
        body:  "Download your results as a <b>PDF report</b> for record-keeping or to share your progress.",
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
    return "senyasalin_tutorial_dismissed" + (userId ? "_" + userId : "");
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
    // Auto-show after a brief delay so the page finishes rendering
    setTimeout(() => start(pageName, false), 600);
  }

  return { init, start, resetAll };
})();
