document.addEventListener("DOMContentLoaded", function () {

    const palette = __md_get("__palette")
    const useDark = palette && typeof palette.color === "object" && palette.color.scheme === "slate"
    const theme = useDark ? "dark-theme" : "light-default";

    const colorList = [
        "#22c55e",
        "#14b8a6",
        "#ef4444",
        "#eab308",
        "#8b5cf6",
        "#f97316",
        "#3b82f6",
    ]

    const logoSrc = (document.querySelector('link[rel="icon"]') || {}).href || '';
    const authorCache = {};

    const repoCards = document.querySelectorAll(".repo-card");
    const labelsAll = Array
        .from(repoCards)
        .flatMap((element) => (element.getAttribute('data-labels') || '').split(','))
        .map(label => label.trim())
        .filter(label => label !== '');
    const uniqueLabels = [...new Set(labelsAll)];

    const labelToColor = uniqueLabels.reduce((map, label, index) => {
        map[label] = colorList[index % colorList.length];
        return map;
    }, {});


    async function renderCard(element, elementIndex) {
        const name = element.getAttribute('data-name') || '';
        const description = element.getAttribute('data-description') || '';
        const labels = element.getAttribute('data-labels') || '';
        const version = element.getAttribute('data-version') || '';
        const authors = element.getAttribute('data-author') || '';

        const labelHTML = labels ? labels.split(',').filter(label => label !== '').map((label, index) => {
            const color = labelToColor[label.trim()];
            return `
            <span
                class="label non-selectable-text"
                style="background-color: ${color}"
            >
                ${label.trim()}
            </span>
        `;
        }).join(' ') : '';

        const authorArray = authors ? authors.split(',').filter(a => a.trim()) : [];
        const authorDataArray = await Promise.all(authorArray.map(async (author) => {
            const login = author.trim();
            if (authorCache[login]) return authorCache[login];
            try {
                const response = await fetch(`https://api.github.com/users/${login}`);
                if (!response.ok) return { login, avatar_url: `https://github.com/${login}.png` };
                const data = await response.json();
                authorCache[login] = data;
                return data;
            } catch {
                return { login, avatar_url: `https://github.com/${login}.png` };
            }
        }));

        let authorAvatarsHTML = authorDataArray.map((authorData, index) => {
            const marginLeft = index === 0 ? '0' : '-10px';
            return `
            <div
                class="author-container"
                data-login="${authorData.login}-${elementIndex}"
                style="margin-left: ${marginLeft};"
            >
                <a
                    href="https://github.com/${authorData.login}"
                    target="_blank"
                    rel="noopener noreferrer"
                    style="line-height: 0;"
                >
                    <img
                        class="author-avatar"
                        src="${authorData.avatar_url}"
                        alt="${authorData.login}'s avatar"
                    >
                </a>
            </div>
        `;
        }).join('');

        let authorNamesHTML = authorDataArray.map(
            authorData => `
            <span
                class="author-name"
                data-login="${authorData.login}-${elementIndex}"
            >
            <a href="https://github.com/${authorData.login}" target="_blank" rel="noopener noreferrer">
                ${authorData.login}
            </a>
        </span>`
        ).join(',&nbsp;');

        let authorsHTML = `
        <div class="authors" style="margin: 0;">
            ${authorAvatarsHTML}
            <div class="author-names">${authorNamesHTML}</div>
        </div>
        `;

        const rawHTML = `
            <div style="
                display: grid !important;
                grid-template-rows: auto;
                height: 100%;
            font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif,Apple Color Emoji,Segoe UI Emoji; font-size: 14px; line-height: 1.5;">
            <div style="display: flex; align-items: center;">
                <span style="font-weight: 700; font-size: 1rem;">
                ${name}
                </span>
            </div>
            ${description ? `<p style="font-size: 0.85rem; opacity: 0.75; margin: 0.5rem 0 0;">${description}</p>` : ''}
            ${authorsHTML}
            <div style="font-size: 12px; display: grid; grid-template-columns: auto 3fr; justify-content: space-between; gap: 1rem;">
                <div style="display: flex; align-items: center;">
                <img src="${logoSrc}" aria-label="rf-detr" width="20" height="20" role="img" />
                &nbsp;
                <span style="margin-left: 4px">${version}</span>
                </div>
                <div style="display: flex; align-items: center; flex-wrap: wrap; align-content: right; gap: 0.1rem;">
                ${labelHTML}
                </div>
            </div>
        </div>
        `;

        element.innerHTML = DOMPurify.sanitize(rawHTML);

        element.querySelectorAll('.author-name').forEach(nameEl => {
            nameEl.addEventListener('mouseenter', function () {
                const login = this.getAttribute('data-login');
                element.querySelector(`.author-container[data-login="${login}"]`).classList.add('hover');
            });

            nameEl.addEventListener('mouseleave', function () {
                const login = this.getAttribute('data-login');
                element.querySelector(`.author-container[data-login="${login}"]`).classList.remove('hover');
            });
        });
    }
    repoCards.forEach((element, index) => {
        renderCard(element, index);
    });
})
