import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

const VIEWPORT_PX = [1280, 720];
const CANVAS_CSS_SIZES_PX = {
    'c': VIEWPORT_PX,
    'obs-canvas': [390, 390],
    'agent-view-canvas': [640, 360],
};
const DEFAULT_CSS_SIZE_PX = [320, 180];
const LOAD_TIMEOUT_MS = 180000;
const POLL_INTERVAL_MS = 20;
const MAX_ANIMATION_FRAME_ROUNDS = 8;
const POSITION_MATCH_TOLERANCE_M = 1e-9;
const EMPTY_CLICK_MIN_CLEARANCE_M = 3;
const CONTEXT_DEFAULTS = {
    fillStyle: '#000000',
    strokeStyle: '#000000',
    lineWidth: 1,
    globalAlpha: 1,
    font: '10px sans-serif',
    textAlign: 'start',
    textBaseline: 'alphabetic',
    lineCap: 'butt',
    lineJoin: 'miter',
    miterLimit: 10,
    lineDashOffset: 0,
    globalCompositeOperation: 'source-over',
    shadowBlur: 0,
    shadowColor: 'rgba(0,0,0,0)',
    shadowOffsetX: 0,
    shadowOffsetY: 0,
    imageSmoothingEnabled: true,
    filter: 'none',
    direction: 'ltr',
    letterSpacing: '0px',
};
const gradient = () => ({addColorStop() {}});
const CONTEXT_RETURNS = {
    createLinearGradient: gradient,
    createRadialGradient: gradient,
    createConicGradient: gradient,
    createPattern: () => ({setTransform() {}}),
    measureText: (text) => {
        const width = String(text).length * 6;
        return {width, actualBoundingBoxAscent: 7, actualBoundingBoxDescent: 2, actualBoundingBoxLeft: 0, actualBoundingBoxRight: width};
    },
    getTransform: () => ({a: 1, b: 0, c: 0, d: 1, e: 0, f: 0, inverse() { return this; }}),
    isPointInPath: () => false,
    isPointInStroke: () => false,
    getImageData: (x, y, w, h) => ({width: w, height: h, data: new Uint8ClampedArray(Math.max(0, w * h * 4))}),
    createImageData: (w, h) => ({width: w, height: h, data: new Uint8ClampedArray(Math.max(0, w * h * 4))}),
    getContextAttributes: () => ({alpha: true}),
};

let activePage = null;

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function describeError(error) {
    return error && error.stack ? String(error.stack) : String(error);
}

function plain(value) {
    return value === undefined ? null : JSON.parse(JSON.stringify(value));
}

class HarnessPath2D {
    constructor(source) {
        this.ops = source && source.ops ? source.ops.slice() : [];
    }

    addPath(other) {
        if (other && other.ops) this.ops.push(...other.ops);
    }
}
for (const method of ['moveTo', 'lineTo', 'closePath', 'arc', 'arcTo', 'rect', 'roundRect', 'ellipse', 'bezierCurveTo', 'quadraticCurveTo']) {
    HarnessPath2D.prototype[method] = function recordPathOp() {
        this.ops.push(method);
    };
}

class FakeClassList {
    constructor() {
        this.names = new Set();
    }

    add(...names) {
        names.forEach((name) => this.names.add(name));
    }

    remove(...names) {
        names.forEach((name) => this.names.delete(name));
    }

    toggle(name, force) {
        const on = force === undefined ? !this.names.has(name) : Boolean(force);
        if (on) this.names.add(name);
        else this.names.delete(name);
        return on;
    }

    contains(name) {
        return this.names.has(name);
    }

    replace(oldName, newName) {
        this.names.delete(oldName);
        this.names.add(newName);
    }
}

function makeStyle() {
    const style = {};
    Object.defineProperties(style, {
        setProperty: {value(name, value) { style[name] = value; }},
        removeProperty: {value(name) { delete style[name]; }},
        getPropertyValue: {value(name) { return style[name] || ''; }},
    });
    return style;
}

function makeRecordingContext(canvas) {
    const state = {...CONTEXT_DEFAULTS, canvas};
    const calls = [];
    let lineDash = [];
    return new Proxy({}, {
        get(_, property) {
            if (property === '__calls') return calls;
            if (property === '__reset') return () => { calls.length = 0; };
            if (typeof property === 'symbol' || property === 'then') return undefined;
            if (Object.prototype.hasOwnProperty.call(state, property)) return state[property];
            if (property === 'getLineDash') return () => lineDash.slice();
            return (...args) => {
                if (property === 'setLineDash') lineDash = Array.from(args[0] || []);
                calls.push({name: property, args});
                const producer = CONTEXT_RETURNS[property];
                return producer ? producer(...args) : undefined;
            };
        },
        set(_, property, value) {
            state[property] = value;
            return true;
        },
        has() {
            return true;
        },
    });
}

class FakeElement {
    constructor(page, id, tagName) {
        this.page = page;
        this.id = id;
        this.tagName = tagName;
        this.nodeName = tagName;
        this.style = makeStyle();
        this.classList = new FakeClassList();
        this.dataset = {};
        this.attributes = {};
        this.textContent = '';
        this.innerHTML = '';
        this.innerText = '';
        this.value = '';
        this.title = '';
        this.hidden = false;
        this.disabled = false;
        this.children = [];
        this.childNodes = [];
        this.parentElement = null;
        this.parentNode = null;
        this.listeners = {};
        this.width = 300;
        this.height = 150;
        this.scrollTop = 0;
        this.scrollLeft = 0;
        this.removed = false;
        this.context = null;
    }

    cssSize() {
        return CANVAS_CSS_SIZES_PX[this.id] || DEFAULT_CSS_SIZE_PX;
    }

    get offsetWidth() { return this.cssSize()[0]; }
    get offsetHeight() { return this.cssSize()[1]; }
    get clientWidth() { return this.cssSize()[0]; }
    get clientHeight() { return this.cssSize()[1]; }
    get scrollWidth() { return this.cssSize()[0]; }
    get scrollHeight() { return this.cssSize()[1]; }
    get firstChild() { return this.children[0] || null; }
    get lastChild() { return this.children[this.children.length - 1] || null; }

    getBoundingClientRect() {
        const [width, height] = this.cssSize();
        return {left: 0, top: 0, x: 0, y: 0, width, height, right: width, bottom: height};
    }

    getClientRects() {
        return [this.getBoundingClientRect()];
    }

    getContext() {
        if (!this.context) this.context = makeRecordingContext(this);
        return this.context;
    }

    addEventListener(type, listener) {
        (this.listeners[type] ||= []).push(listener);
    }

    removeEventListener(type, listener) {
        const listeners = this.listeners[type] || [];
        const index = listeners.indexOf(listener);
        if (index >= 0) listeners.splice(index, 1);
    }

    dispatchEvent(event) {
        for (const listener of this.listeners[event.type] || []) listener(event);
        return true;
    }

    querySelector() {
        return new FakeElement(this.page, null, 'DIV');
    }

    querySelectorAll() {
        return [];
    }

    getElementsByClassName() {
        return [];
    }

    closest() {
        return null;
    }

    matches() {
        return false;
    }

    contains(other) {
        return other === this;
    }

    remove() {
        this.removed = true;
    }

    appendChild(child) {
        this.children.push(child);
        return child;
    }

    append(...items) {
        this.children.push(...items);
    }

    prepend(...items) {
        this.children.unshift(...items);
    }

    insertBefore(child) {
        this.children.unshift(child);
        return child;
    }

    removeChild(child) {
        this.children = this.children.filter((item) => item !== child);
        return child;
    }

    replaceChildren(...items) {
        this.children = items;
    }

    setAttribute(name, value) {
        this.attributes[name] = String(value);
    }

    getAttribute(name) {
        return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null;
    }

    hasAttribute(name) {
        return Object.prototype.hasOwnProperty.call(this.attributes, name);
    }

    removeAttribute(name) {
        delete this.attributes[name];
    }

    toggleAttribute(name, force) {
        const on = force === undefined ? !this.hasAttribute(name) : Boolean(force);
        if (on) this.attributes[name] = '';
        else delete this.attributes[name];
        return on;
    }

    focus() {}
    blur() {}
    scrollIntoView() {}
    setPointerCapture() {}
    releasePointerCapture() {}

    animate() {
        return {cancel() {}, finished: Promise.resolve()};
    }

    click() {
        const event = {type: 'click', target: this, currentTarget: this, preventDefault() {}, stopPropagation() {}};
        this.dispatchEvent(event);
        if (typeof this.onclick === 'function') this.onclick(event);
    }
}

function camelCase(name) {
    return name.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
}

function makeWorkerClass(page) {
    return class HarnessWorker {
        constructor(url) {
            this.onmessage = null;
            this.onerror = null;
            this.terminated = false;
            this.ready = false;
            this.inbox = [];
            const worker = this;
            this.scope = {
                onmessage: null,
                postMessage(data) {
                    setTimeout(() => {
                        if (!worker.terminated && worker.onmessage) worker.onmessage({data});
                    }, 0);
                },
                close() {},
            };
            const blob = page.blobs.get(url);
            if (!blob) {
                page.errors.push(`Worker: unknown blob url ${url}`);
                return;
            }
            blob.text().then((source) => {
                new Function('self', 'postMessage', source)(this.scope, this.scope.postMessage);
                this.ready = true;
                this.flush();
            }).catch((error) => {
                page.errors.push(`Worker load: ${describeError(error)}`);
                if (this.onerror) this.onerror({message: String(error)});
            });
        }

        postMessage(data) {
            this.inbox.push(data);
            if (this.ready) this.flush();
        }

        flush() {
            while (this.inbox.length) {
                const data = this.inbox.shift();
                setTimeout(() => {
                    if (this.terminated) return;
                    Promise.resolve().then(() => this.scope.onmessage({data})).catch((error) => {
                        page.errors.push(`Worker message: ${describeError(error)}`);
                        if (this.onerror) this.onerror({message: String(error)});
                    });
                }, 0);
            }
        }

        addEventListener(type, listener) {
            if (type === 'message') this.onmessage = listener;
            if (type === 'error') this.onerror = listener;
        }

        terminate() {
            this.terminated = true;
        }
    };
}

function makeStorage() {
    const values = new Map();
    return {
        getItem: (key) => (values.has(key) ? values.get(key) : null),
        setItem: (key, value) => values.set(key, String(value)),
        removeItem: (key) => values.delete(key),
        clear: () => values.clear(),
    };
}

class Page {
    constructor(name, htmlPath) {
        this.name = name;
        this.htmlPath = htmlPath;
        this.html = fs.readFileSync(htmlPath, 'utf8');
        this.errors = [];
        this.animationFrames = [];
        this.documentListeners = {};
        this.windowListeners = {};
        this.elements = new Map();
        this.payloadNodes = [];
        this.scripts = [];
        this.blobs = new Map();
        this.blobCounter = 0;
        this.loaded = false;
        this.parse();
    }

    parse() {
        const scriptPattern = /<script\b([^>]*)>([\s\S]*?)<\/script>/g;
        const markup = this.html.replace(scriptPattern, (_, attrs) => `<script${attrs}></script>`);
        for (const match of markup.matchAll(/<([a-zA-Z][a-zA-Z0-9]*)\b([^>]*)>/g)) {
            const attrs = match[2];
            const idMatch = /\bid="([^"]+)"/.exec(attrs);
            if (!idMatch) continue;
            const element = new FakeElement(this, idMatch[1], match[1].toUpperCase());
            const classMatch = /\bclass="([^"]*)"/.exec(attrs);
            if (classMatch) classMatch[1].split(/\s+/).filter(Boolean).forEach((name) => element.classList.add(name));
            for (const dataMatch of attrs.matchAll(/\bdata-([\w-]+)="([^"]*)"/g)) element.dataset[camelCase(dataMatch[1])] = dataMatch[2];
            this.elements.set(idMatch[1], element);
        }
        for (const match of this.html.matchAll(scriptPattern)) {
            const attrs = match[1];
            const body = match[2];
            if (/class="payload-chunk"/.test(attrs)) {
                const node = new FakeElement(this, null, 'SCRIPT');
                node.textContent = body;
                this.payloadNodes.push(node);
                continue;
            }
            const idMatch = /\bid="([^"]+)"/.exec(attrs);
            if (idMatch && this.elements.has(idMatch[1])) this.elements.get(idMatch[1]).textContent = body;
            const typeMatch = /\btype="([^"]*)"/.exec(attrs);
            if (!typeMatch || /javascript|module/.test(typeMatch[1])) this.scripts.push(body);
        }
    }

    query(selector) {
        if (selector === '.payload-chunk') return this.payloadNodes.filter((node) => !node.removed);
        if (selector.startsWith('#')) {
            const element = this.elements.get(selector.slice(1));
            return element ? [element] : [];
        }
        if (/^\.[\w-]+$/.test(selector)) {
            const className = selector.slice(1);
            return [...this.elements.values()].filter((element) => element.classList.contains(className));
        }
        return [];
    }

    makeDocument() {
        const page = this;
        const body = new FakeElement(page, null, 'BODY');
        return {
            documentElement: new FakeElement(page, null, 'HTML'),
            body,
            head: new FakeElement(page, null, 'HEAD'),
            readyState: 'complete',
            hidden: false,
            visibilityState: 'visible',
            activeElement: body,
            getElementById: (id) => page.elements.get(id) || null,
            querySelector: (selector) => page.query(selector)[0] || null,
            querySelectorAll: (selector) => page.query(selector),
            getElementsByClassName: (name) => page.query(`.${name}`),
            createElement: (tag) => new FakeElement(page, null, String(tag).toUpperCase()),
            createElementNS: (_, tag) => new FakeElement(page, null, String(tag).toUpperCase()),
            createTextNode: (text) => ({textContent: String(text)}),
            addEventListener: (type, listener) => (page.documentListeners[type] ||= []).push(listener),
            removeEventListener: (type, listener) => {
                const listeners = page.documentListeners[type] || [];
                const index = listeners.indexOf(listener);
                if (index >= 0) listeners.splice(index, 1);
            },
        };
    }

    async load() {
        activePage = this;
        const page = this;
        this.document = this.makeDocument();
        class HarnessResizeObserver {
            constructor(callback) {
                this.callback = callback;
            }

            observe(target) {
                setTimeout(() => {
                    try {
                        this.callback([{target, contentRect: target.getBoundingClientRect()}], this);
                    } catch (error) {
                        page.errors.push(`ResizeObserver callback: ${describeError(error)}`);
                    }
                }, 0);
            }

            unobserve() {}
            disconnect() {}
        }
        class HarnessURL extends URL {}
        HarnessURL.createObjectURL = (blob) => {
            page.blobCounter += 1;
            const url = `blob:harness/${page.blobCounter}`;
            page.blobs.set(url, blob);
            return url;
        };
        HarnessURL.revokeObjectURL = () => {};
        const sandbox = {
            console: {
                log() {},
                info() {},
                debug() {},
                warn() {},
                error: (...items) => page.errors.push(`console.error: ${items.map(describeError).join(' ')}`),
            },
            document: this.document,
            navigator: {userAgent: 'replay-viewer-harness', language: 'en', clipboard: {writeText: async () => {}}},
            location: {href: `file://${this.htmlPath}`, search: '', hash: '', pathname: this.htmlPath},
            innerWidth: VIEWPORT_PX[0],
            innerHeight: VIEWPORT_PX[1],
            devicePixelRatio: 1,
            getComputedStyle: () => ({getPropertyValue: () => '#888888'}),
            requestAnimationFrame: (callback) => {
                page.animationFrames.push(callback);
                return page.animationFrames.length;
            },
            cancelAnimationFrame: () => {},
            ResizeObserver: HarnessResizeObserver,
            MutationObserver: class { observe() {} disconnect() {} takeRecords() { return []; } },
            IntersectionObserver: class { observe() {} unobserve() {} disconnect() {} },
            matchMedia: () => ({matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}}),
            localStorage: makeStorage(),
            sessionStorage: makeStorage(),
            setTimeout,
            clearTimeout,
            setInterval,
            clearInterval,
            queueMicrotask,
            structuredClone,
            performance,
            TextDecoder,
            TextEncoder,
            Blob,
            Response,
            DecompressionStream,
            atob,
            btoa,
            URL: HarnessURL,
            Worker: makeWorkerClass(page),
            Path2D: HarnessPath2D,
            Image: class { set src(_) { setTimeout(() => this.onload && this.onload(), 0); } },
            addEventListener: (type, listener) => (page.windowListeners[type] ||= []).push(listener),
            removeEventListener: () => {},
            dispatchEvent: () => true,
            scrollTo() {},
            alert() {},
            confirm: () => true,
            open() {},
        };
        sandbox.window = sandbox;
        sandbox.self = sandbox;
        sandbox.top = sandbox;
        sandbox.parent = sandbox;
        this.context = vm.createContext(sandbox);
        for (const [scriptIndex, script] of this.scripts.entries()) {
            try {
                vm.runInContext(script, this.context, {filename: `${path.basename(this.htmlPath)}#script${scriptIndex}`});
            } catch (error) {
                this.errors.push(`script: ${describeError(error)}`);
            }
        }
        const deadline = Date.now() + LOAD_TIMEOUT_MS;
        while (Date.now() < deadline) {
            await this.settle();
            if (this.isReady()) {
                this.loaded = true;
                break;
            }
            await sleep(POLL_INTERVAL_MS);
        }
        await this.settle();
        if (!this.loaded) {
            const loadText = this.elements.get('load-text');
            this.errors.push(`page did not finish loading: load-text=${loadText ? loadText.textContent : '?'}`);
        }
    }

    isReady() {
        try {
            return this.run('typeof H !== "undefined" && !!H && typeof roadSegCount !== "undefined" && roadSegCount > 0') === true;
        } catch (error) {
            return false;
        }
    }

    async settle() {
        for (let round = 0; round < MAX_ANIMATION_FRAME_ROUNDS; round++) {
            await sleep(0);
            const callbacks = this.animationFrames.splice(0);
            if (!callbacks.length) break;
            for (const callback of callbacks) {
                try {
                    callback(performance.now());
                } catch (error) {
                    this.errors.push(`requestAnimationFrame: ${describeError(error)}`);
                }
            }
        }
        await sleep(0);
    }

    run(expression) {
        return vm.runInContext(expression, this.context);
    }

    setGlobal(name, value) {
        this.context[name] = value;
    }

    async select(agentId, frame) {
        this.run(`followedId = ${agentId === null ? 'null' : agentId}; step = ${frame}; play = false;`);
        this.run('draw(true)');
        await this.settle();
    }

    async keydown(code, key) {
        const event = {
            type: 'keydown',
            code,
            key,
            target: this.document.body,
            repeat: false,
            shiftKey: false,
            ctrlKey: false,
            metaKey: false,
            altKey: false,
            preventDefault() {},
            stopPropagation() {},
        };
        for (const listener of [...(this.documentListeners.keydown || []), ...(this.windowListeners.keydown || [])]) listener(event);
        await this.settle();
    }

    async mousedown(clientX, clientY) {
        const canvas = this.elements.get('c');
        const event = {
            type: 'mousedown',
            clientX,
            clientY,
            offsetX: clientX,
            offsetY: clientY,
            button: 0,
            buttons: 1,
            target: canvas,
            currentTarget: canvas,
            preventDefault() {},
            stopPropagation() {},
        };
        if (typeof canvas.onmousedown === 'function') canvas.onmousedown(event);
        canvas.dispatchEvent(event);
        if (typeof canvas.onpointerdown === 'function') canvas.onpointerdown({...event, type: 'pointerdown'});
        canvas.dispatchEvent({...event, type: 'pointerdown'});
        const upEvent = {...event, type: 'mouseup', buttons: 0};
        if (typeof this.context.onmouseup === 'function') this.context.onmouseup(upEvent);
        for (const listener of this.windowListeners.mouseup || []) listener(upEvent);
        for (const listener of this.windowListeners.pointerup || []) listener({...upEvent, type: 'pointerup'});
        await this.settle();
    }

    pill() {
        const pill = this.elements.get('view-pill');
        if (!pill) return {exists: false, text: '', visible: false};
        const text = String(pill.textContent || pill.innerText || pill.innerHTML || '');
        const visible = pill.style.display !== 'none' && !pill.hidden && !pill.classList.contains('hidden') && text.length > 0;
        return {exists: true, text, visible};
    }

    observedOnly() {
        return this.run('observedOnly');
    }

    allText() {
        const texts = [];
        for (const element of this.elements.values()) texts.push(String(element.textContent || ''), String(element.innerHTML || ''));
        for (const element of this.elements.values()) {
            if (!element.context) continue;
            for (const call of element.context.__calls) {
                if (call.name === 'fillText' || call.name === 'strokeText') texts.push(String(call.args[0]));
            }
        }
        return texts.join('\n');
    }
}

function countCalls(calls) {
    const counts = {};
    for (const call of calls) counts[call.name] = (counts[call.name] || 0) + 1;
    return counts;
}

function drawnAgentIndices(calls, agents) {
    const translations = calls.filter((call) => call.name === 'translate' && call.args.length >= 2);
    return agents
        .filter((agent) => translations.some((call) => Math.abs(call.args[0] - agent.x) <= POSITION_MATCH_TOLERANCE_M && Math.abs(call.args[1] - agent.y) <= POSITION_MATCH_TOLERANCE_M))
        .map((agent) => agent.idx);
}

function strokedPaths(calls) {
    return calls.filter((call) => call.name === 'stroke' && call.args[0] instanceof HarnessPath2D).map((call) => call.args[0]);
}

async function openPage(report, name, htmlPath) {
    const page = new Page(name, htmlPath);
    await page.load();
    report.pages[name] = {loaded: page.loaded, errors: page.errors};
    return page;
}

async function runStep(report, name, body) {
    try {
        report.data[name] = await body();
    } catch (error) {
        report.stepErrors[name] = describeError(error);
    }
}

async function resetObserved(page, agentId, frame) {
    await page.select(agentId, frame);
    if (page.observedOnly()) await page.keydown('KeyV', 'v');
    if (page.observedOnly()) page.run('observedOnly = false');
    await page.select(agentId, frame);
}

function emptyClickPoint(page, config) {
    const camera = page.run('({x: cam.x, y: cam.y, z: cam.z})');
    const canvas = page.elements.get('c');
    const corners = [[1, 1], [canvas.width - 1, 1], [1, canvas.height - 1], [canvas.width - 1, canvas.height - 1]];
    for (const [clientX, clientY] of corners) {
        const worldX = (clientX - canvas.width / 2) / camera.z + camera.x;
        const worldY = (clientY - canvas.height / 2) / -camera.z + camera.y;
        const clear = config.validAgents.every((agent) => Math.hypot(worldX - agent.x, worldY - agent.y) > Math.max(agent.l, EMPTY_CLICK_MIN_CLEARANCE_M));
        if (clear) return {clientX, clientY, worldX, worldY};
    }
    return null;
}

async function checkMainPage(report, config) {
    const page = await openPage(report, 'main', config.mainHtml);
    const frame = config.frame;
    const targetIdx = config.targetIdx;
    const targetId = config.targetId;
    if (!page.loaded) return;

    await runStep(report, 'headerVersion', () => page.run('H.replay_format_version'));

    await runStep(report, 'camera', async () => {
        await page.select(targetId, frame);
        const result = {};
        for (const [preset, reference] of Object.entries(config.cameraRefs)) {
            const camera = page.run(`agentViewCamera(agentAt(${frame}, ${targetIdx}), ${JSON.stringify(preset)}, ${config.cameraWidthPx}, ${config.cameraHeightPx})`);
            page.setGlobal('__harnessCamera', camera);
            const projected = reference.points.map((point) => Array.from(page.run(`camProject(__harnessCamera, ${point[0]}, ${point[1]}, ${point[2]})`)));
            result[preset] = {camera: plain(camera), projected};
        }
        return result;
    });

    await runStep(report, 'decode', async () => {
        await page.select(targetId, frame);
        return plain(page.run(`decodeObservedWorld(${frame}, agentAt(${frame}, ${targetIdx}))`));
    });

    await runStep(report, 'agentView', async () => {
        await resetObserved(page, targetId, frame);
        const canvas = page.elements.get('agent-view-canvas');
        const context = canvas.getContext('2d');
        const box = page.elements.get('agent-view-box');
        const hud = page.elements.get('hud-telemetry');
        const onBefore = page.run('agentViewOn');
        context.__reset();
        page.run('toggleAgentView()');
        await page.settle();
        page.run('draw(true)');
        await page.settle();
        const openCalls = countCalls(context.__calls);
        const onAfterOpen = page.run('agentViewOn');
        const boxDisplay = box.style.display;
        const presets = [page.run('agentViewPreset')];
        page.run('cycleAgentViewPreset()');
        await page.settle();
        presets.push(page.run('agentViewPreset'));
        context.__reset();
        page.run('draw(true)');
        await page.settle();
        const driverCallCount = context.__calls.length;
        page.run('cycleAgentViewPreset()');
        await page.settle();
        presets.push(page.run('agentViewPreset'));
        const widthBefore = String(hud.style.width || '');
        page.run('toggleAgentViewSize()');
        await page.settle();
        const widthExpanded = String(hud.style.width || '');
        const expandedClassAfterToggle = hud.classList.contains('expanded');
        page.run('toggleAgentViewSize()');
        await page.settle();
        const widthCollapsed = String(hud.style.width || '');
        const expandedClassAfterSecondToggle = hud.classList.contains('expanded');

        context.__reset();
        page.run('draw(true)');
        const fillsWithoutV = countCalls(context.__calls).fill || 0;
        await page.keydown('KeyV', 'v');
        const observedForAgentView = page.observedOnly();
        context.__reset();
        page.run('draw(true)');
        await page.settle();
        const fillsWithV = countCalls(context.__calls).fill || 0;
        await page.keydown('KeyV', 'v');

        page.run('toggleAgentView()');
        await page.settle();
        const onAfterClose = page.run('agentViewOn');
        context.__reset();
        page.run('draw(true)');
        await page.settle();
        return {
            onBefore,
            onAfterOpen,
            onAfterClose,
            boxExists: Boolean(box),
            boxDisplay,
            canvasWidth: canvas.width,
            canvasHeight: canvas.height,
            openCalls,
            driverCallCount,
            callsAfterClose: context.__calls.length,
            presets,
            widthBefore,
            widthExpanded,
            widthCollapsed,
            expandedClassAfterToggle,
            expandedClassAfterSecondToggle,
            fillsWithoutV,
            fillsWithV,
            observedForAgentView,
            errorsSoFar: page.errors.length,
        };
    });

    await runStep(report, 'observedOnly', async () => {
        await resetObserved(page, targetId, frame);
        const mainContext = page.elements.get('c').getContext('2d');
        const mapPaths = page.run('[paths[0], paths[1], paths[2]]');
        const lanePaths = page.run('Array.from(roadPathsById.values())');
        mainContext.__reset();
        page.run('draw(true)');
        const offCalls = mainContext.__calls.slice();
        const offBefore = page.observedOnly();
        await page.keydown('KeyV', 'v');
        const onAfterV = page.observedOnly();
        const pillWhenOn = page.pill();
        mainContext.__reset();
        page.run('draw(true)');
        await page.settle();
        const onCalls = mainContext.__calls.slice();
        const offStroked = strokedPaths(offCalls);
        const onStroked = strokedPaths(onCalls);
        await page.keydown('KeyV', 'v');
        const offAfterSecondV = page.observedOnly();
        const pillAfterSecondV = page.pill();
        return {
            offBefore,
            onAfterV,
            offAfterSecondV,
            pillWhenOn,
            pillAfterSecondV,
            drawnWithoutV: drawnAgentIndices(offCalls, config.validAgents),
            drawnWithV: drawnAgentIndices(onCalls, config.validAgents),
            mapPathOpCounts: mapPaths.map((mapPath) => mapPath.ops.length),
            mapPathsStrokedWithoutV: mapPaths.map((mapPath) => offStroked.includes(mapPath)),
            mapPathsStrokedWithV: mapPaths.map((mapPath) => onStroked.includes(mapPath)),
            lanePathsStrokedWithV: onStroked.filter((stroked) => lanePaths.includes(stroked)).length,
            strokeCountWithoutV: countCalls(offCalls).stroke || 0,
            strokeCountWithV: countCalls(onCalls).stroke || 0,
        };
    });

    await runStep(report, 'escapeTurnsVOff', async () => {
        await resetObserved(page, targetId, frame);
        await page.keydown('KeyV', 'v');
        const onAfterV = page.observedOnly();
        await page.keydown('Escape', 'Escape');
        return {onAfterV, afterEscape: page.observedOnly()};
    });

    await runStep(report, 'emptyClickTurnsVOff', async () => {
        await resetObserved(page, targetId, frame);
        await page.keydown('KeyV', 'v');
        const onAfterV = page.observedOnly();
        const point = emptyClickPoint(page, config);
        if (point === null) return {onAfterV, point: null, afterClick: null};
        await page.mousedown(point.clientX, point.clientY);
        return {onAfterV, point, afterClick: page.observedOnly()};
    });

    await runStep(report, 'noSelection', async () => {
        await resetObserved(page, null, frame);
        await page.keydown('KeyV', 'v');
        return {afterV: page.observedOnly()};
    });

    await runStep(report, 'absentAgent', async () => {
        await resetObserved(page, config.absentId, frame);
        const before = page.observedOnly();
        await page.keydown('KeyV', 'v');
        page.run('draw(true)');
        await page.settle();
        return {before, afterV: page.observedOnly(), pill: page.pill()};
    });

    report.pages.main.errors = page.errors;
}

async function checkNoObservationPage(report, config) {
    const page = await openPage(report, 'noObs', config.noObsHtml);
    if (!page.loaded) return;
    await runStep(report, 'noObs', async () => {
        await resetObserved(page, config.targetId, config.frame);
        const decode = plain(page.run(`decodeObservedWorld(${config.frame}, agentAt(${config.frame}, ${config.targetIdx}))`));
        await page.keydown('KeyV', 'v');
        page.run('draw(true)');
        await page.settle();
        return {decode, afterV: page.observedOnly(), pill: page.pill()};
    });
    report.pages.noObs.errors = page.errors;
}

async function checkBlindPage(report, config) {
    const page = await openPage(report, 'blind', config.blindHtml);
    if (!page.loaded) return;
    await runStep(report, 'blind', async () => {
        await resetObserved(page, config.targetId, config.frame);
        await page.keydown('KeyV', 'v');
        page.run('draw(true)');
        await page.settle();
        const decode = plain(page.run(`decodeObservedWorld(${config.frame}, agentAt(${config.frame}, ${config.targetIdx}))`));
        return {afterV: page.observedOnly(), pill: page.pill(), partnerBlind: decode ? decode.partnerBlind : null};
    });
    await runStep(report, 'staticAgent', async () => {
        await resetObserved(page, config.staticId, config.frame);
        const slot = page.run(`agentAt(${config.frame}, ${config.staticIdx}).slot`);
        await page.keydown('KeyV', 'v');
        page.run('draw(true)');
        await page.settle();
        return {slot, afterV: page.observedOnly(), pill: page.pill()};
    });
    report.pages.blind.errors = page.errors;
}

async function checkOldFormatPage(report, config) {
    const page = await openPage(report, 'oldFormat', config.oldFormatHtml);
    if (!page.loaded) return;
    await runStep(report, 'oldFormat', async () => {
        await resetObserved(page, config.targetId, config.frame);
        page.run('draw(true)');
        await page.settle();
        const text = page.allText();
        await page.keydown('KeyV', 'v');
        return {mentionsOldFormat: text.includes('old format'), afterV: page.observedOnly()};
    });
    report.pages.oldFormat.errors = page.errors;
}

async function main() {
    const config = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
    const report = {pages: {}, data: {}, stepErrors: {}};
    process.on('unhandledRejection', (error) => {
        if (activePage) activePage.errors.push(`unhandledRejection: ${describeError(error)}`);
    });
    process.on('uncaughtException', (error) => {
        if (activePage) activePage.errors.push(`uncaughtException: ${describeError(error)}`);
    });
    await checkMainPage(report, config);
    await checkNoObservationPage(report, config);
    await checkBlindPage(report, config);
    await checkOldFormatPage(report, config);
    fs.writeFileSync(config.reportPath, JSON.stringify(report));
    process.exit(0);
}

main().catch((error) => {
    console.error(describeError(error));
    process.exit(2);
});
