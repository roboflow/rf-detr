/*
 * Decides whether the version banner is shown, and publishes its height as a CSS
 * variable.
 *
 * The banner is pinned to the top of the viewport, so the sticky header and
 * sidebars have to start below it instead of scrolling underneath. Its height
 * cannot be hardcoded: the banner wraps at different viewport widths and stays
 * hidden entirely on the latest release docs.
 *
 * The reveal is decided here rather than left to Material. Material tests
 * `extra.version.default` ("latest") against each versions.json entry's aliases
 * plus its version name; this site publishes `latest` as its own mike version
 * instead of as an alias of the current release, so no entry carries that alias
 * and every numbered tree - the newest included - is flagged outdated. Comparing
 * against the highest published release instead keeps the banner off the current
 * release, and still turns it on for that same tree once a newer release ships,
 * without rebuilding it: versions.json is read on every page load.
 */
(() => {
  const banner = document.querySelector("[data-md-component=outdated]");
  if (!banner) {
    return;
  }

  const sidebarLayouts = new WeakMap();
  const sidebarBreakpoints = {
    navigation: window.matchMedia("(min-width: 76.25em)"),
    toc: window.matchMedia("(min-width: 60em)"),
  };

  const syncSidebar = (sidebar, bannerHeight) => {
    const scrollwrap = sidebar.querySelector(".md-sidebar__scrollwrap");
    const breakpoint = sidebarBreakpoints[sidebar.dataset.mdType];
    if (!scrollwrap || !breakpoint) {
      return;
    }

    const layout = sidebarLayouts.get(sidebar) ?? {
      adjustedHeight: "",
      adjustedTop: "",
      baseHeight: "",
      baseTop: "",
    };
    if (sidebar.style.top !== layout.adjustedTop) {
      layout.baseTop = sidebar.style.top;
    }
    if (scrollwrap.style.height !== layout.adjustedHeight) {
      layout.baseHeight = scrollwrap.style.height;
    }

    if (!breakpoint.matches) {
      if (sidebar.style.top === layout.adjustedTop) {
        sidebar.style.top = layout.baseTop;
      }
      if (scrollwrap.style.height === layout.adjustedHeight) {
        scrollwrap.style.height = layout.baseHeight;
      }
      layout.adjustedTop = "";
      layout.adjustedHeight = "";
      sidebarLayouts.set(sidebar, layout);
      return;
    }

    const baseTop = Number.parseFloat(layout.baseTop);
    const baseHeight = Number.parseFloat(layout.baseHeight);
    if (!Number.isFinite(baseTop) || !Number.isFinite(baseHeight)) {
      sidebarLayouts.set(sidebar, layout);
      return;
    }

    layout.adjustedTop = `${baseTop + bannerHeight}px`;
    layout.adjustedHeight = `${baseHeight - bannerHeight}px`;
    if (sidebar.style.top !== layout.adjustedTop) {
      sidebar.style.top = layout.adjustedTop;
    }
    if (scrollwrap.style.height !== layout.adjustedHeight) {
      scrollwrap.style.height = layout.adjustedHeight;
    }
    sidebarLayouts.set(sidebar, layout);
  };

  const syncSidebarLayout = () => {
    const bannerHeight = banner.hidden ? 0 : banner.offsetHeight;
    const sidebars = document.querySelectorAll("[data-md-component=sidebar]");
    for (const sidebar of sidebars) {
      syncSidebar(sidebar, bannerHeight);
    }
  };

  const publishHeight = () => {
    const height = banner.hidden ? 0 : banner.offsetHeight;
    document.documentElement.style.setProperty(
      "--rf-banner-height",
      `${height}px`,
    );
    syncSidebarLayout();
  };

  // Verdict of the version check below: true when this tree is superseded, null
  // while versions.json is still in flight or unreachable. Material runs its own
  // check on the same file and may flip `hidden` afterwards, so once a verdict
  // exists it is re-asserted rather than trusted to stick.
  let outdatedVerdict = null;

  const onHiddenChange = () => {
    if (outdatedVerdict !== null && banner.hidden === outdatedVerdict) {
      banner.hidden = !outdatedVerdict;
      return;
    }
    publishHeight();
  };

  publishHeight();
  // Width changes reflow the text; `hidden` flips once a version check decides
  // the build is outdated, which is after this script runs.
  new ResizeObserver(publishHeight).observe(banner);
  new MutationObserver(onHiddenChange).observe(banner, {
    attributeFilter: ["hidden"],
    attributes: true,
  });
  const sidebarObserver = new MutationObserver(syncSidebarLayout);
  for (const sidebar of document.querySelectorAll("[data-md-component=sidebar]")) {
    sidebarObserver.observe(sidebar, {
      attributeFilter: ["style"],
      attributes: true,
      subtree: true,
    });
  }
  for (const breakpoint of Object.values(sidebarBreakpoints)) {
    breakpoint.addEventListener("change", syncSidebarLayout);
  }

  const isRelease = (name) => /^\d+(\.\d+)*$/.test(name);

  const isNewer = (candidate, incumbent) => {
    const left = candidate.split(".").map(Number);
    const right = incumbent.split(".").map(Number);
    for (let index = 0; index < Math.max(left.length, right.length); index++) {
      const delta = (left[index] ?? 0) - (right[index] ?? 0);
      if (delta !== 0) {
        return delta > 0;
      }
    }
    return false;
  };

  const applyVerdict = (versionBase, outdated) => {
    outdatedVerdict = outdated;
    // Material caches its own verdict under this key and unhides from it, so
    // overwrite it too: a page visited before the fix carries a stale `true`, and
    // agreeing here avoids a visible flash before `onHiddenChange` re-asserts.
    try {
      window.sessionStorage.setItem(
        `${versionBase.pathname}.__outdated`,
        JSON.stringify(outdated),
      );
    } catch (error) {
      // Storage can be unavailable; the attribute is authoritative regardless.
    }
    if (banner.hidden === outdated) {
      banner.hidden = !outdated;
    }
  };

  const configElement = document.getElementById("__config");
  if (configElement) {
    // mkdocs writes a per-page relative base; resolved, it always ends at the
    // version root, so its last segment names this tree and versions.json sits one
    // level above it - the same file and the same scope Material reads.
    const versionBase = new URL(
      JSON.parse(configElement.textContent).base,
      window.location.href,
    );
    const version = versionBase.pathname.replace(/\/$/, "").split("/").pop();

    if (version === "develop") {
      applyVerdict(versionBase, true);
    } else if (isRelease(version)) {
      fetch(new URL("../versions.json", versionBase))
        .then((response) =>
          response.ok ? response.json() : Promise.reject(response.status),
        )
        .then((versions) => {
          const newest = versions
            .map((entry) => entry.version)
            .filter(isRelease)
            .reduce(
              (best, name) => (best === null || isNewer(name, best) ? name : best),
              null,
            );
          applyVerdict(versionBase, newest !== null && newest !== version);
        })
        .catch(() => {
          // versions.json unreachable: leave whatever verdict Material reached.
        });
    }
  }
})();
