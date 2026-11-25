// Tiny trend bot: EMA(8/21) crossover entries plus dip entries inside an
// uptrend, with a momentum-rank filter and quick profit-taking.
//
// How it reads the market (all from the hello message and each batch of
// hourly bars the harness sends; no outside data, no network, no clock):
//   - EMA(8) and EMA(21) of each symbol's closes, seeded as the mean of
//     their first span of closes, updated on every bar.
//   - CCI(14) of the typical price (high + low + close) / 3.
//   - 24-bar momentum rank: each symbol's close now versus 24 bars ago.
//
// Entries (on a bar's close, filled at the next open; at most the stake
// below, sells first then buys, in universe order, 16 orders max per bar):
//   - EMA(8) crosses above EMA(21), or the price is deeply oversold
//     (CCI at or under -50) while EMA(8) still sits above EMA(21); and
//   - the symbol's 24-bar return ranks in the top 4 of the universe
//     (once 24 bars of history exist; before that the filter is off).
// A bar with zero volume signals nothing.
//
// Exits (the whole position): EMA(21) crosses back above EMA(8) and stays
// there, or the position is worth 2% over its cost (quick profit), or 8%
// under it (stop). A refused sell is sent again while its reason holds.
//
// Stake: equity over 7 slots, capped at 25% of equity per symbol and by the
// cash on hand; buys never spend a sell's proceeds from the same reply.
// Orders cap slippage at 100 bps past the close they were decided on, so a
// fill into a liquidity gap is refused rather than filled for nothing.
//
// Deterministic: float64 EMA/CCI on + - * / alone, no clock, no
// environment, no randomness — the same bars give the same orders.

import { BPS, runBars, type Bar, type BarsState, type BuyInput, type SellInput } from './vendor/makefun-bars.ts';

const FAST_SPAN = 8;
const SLOW_SPAN = 21;
const CCI_SPAN = 14;
const RANK_LOOKBACK = 24;
const RANK_TOP = 4;
/** Dip entry: CCI at or under this while the fast EMA is above the slow one. */
const DIP_CCI_AT_MOST = -50;
/** Take profit: worth 101/100 of cost. A 1% win is banked fast, the way the
    dip-buying house baseline does it, then the symbol is bought again on
    the next signal instead of riding a pullback back down. */
const TAKE_NUM = 101n;
const TAKE_DEN = 100n;
/** Stop loss: worth 92/100 of cost. */
const STOP_NUM = 92n;
const STOP_DEN = 100n;
const MAX_SLIPPAGE_BPS = 100;

/** One EMA: closes collected until it starts, then its value. */
interface Ema {
  span: number;
  warm: number[];
  value: number | null;
}

interface Series {
  fast: Ema;
  slow: Ema;
  /** fast - slow after the previous bar, once both EMAs exist. */
  gap: number | null;
  /** fast - slow after this bar. */
  now: number | null;
  /** Sell signal held while EMA(21) stays above EMA(8). */
  down: boolean;
  /** Last 14 typical prices, oldest first, for the CCI. */
  typicals: number[];
  /** Last closes, oldest first, for the 24-bar momentum rank. */
  closes: number[];
}

const series = new Map<string, Series>();

function stepEma(e: Ema, close: number): void {
  if (e.value !== null) {
    e.value += (2 / (e.span + 1)) * (close - e.value);
    return;
  }
  e.warm.push(close);
  if (e.warm.length === e.span) e.value = e.warm.reduce((a, x) => a + x, 0) / e.span;
}

function cciOf(typicals: number[]): number | null {
  if (typicals.length < CCI_SPAN) return null;
  const mean = typicals.reduce((a, x) => a + x, 0) / CCI_SPAN;
  let mad = 0;
  for (const x of typicals) mad += x < mean ? mean - x : x - mean;
  mad /= CCI_SPAN;
  if (mad === 0) return 0;
  return (typicals[typicals.length - 1] - mean) / (0.015 * mad);
}

function pushBar(bar: Bar): Series {
  let s = series.get(bar.symbol);
  if (s === undefined) {
    s = {
      fast: { span: FAST_SPAN, warm: [], value: null },
      slow: { span: SLOW_SPAN, warm: [], value: null },
      gap: null, now: null, down: false, typicals: [], closes: [],
    };
    series.set(bar.symbol, s);
  }
  const c = Number(bar.c);
  stepEma(s.fast, c);
  stepEma(s.slow, c);
  s.gap = s.now;
  s.now = s.fast.value !== null && s.slow.value !== null ? s.fast.value - s.slow.value : null;
  if (s.now === null || s.now >= 0) s.down = false;
  s.typicals.push((Number(bar.h) + Number(bar.l) + c) / 3);
  if (s.typicals.length > CCI_SPAN) s.typicals.shift();
  s.closes.push(c);
  if (s.closes.length > RANK_LOOKBACK + 1) s.closes.shift();
  return s;
}

/** Whether a held position exits on this close. */
function exits(state: BarsState, symbol: string, held: { qty: bigint; cost: bigint }, signal: boolean): boolean {
  if (signal) return true;
  const value = state.value(symbol, held.qty);
  return value * TAKE_DEN >= held.cost * TAKE_NUM || value * STOP_DEN <= held.cost * STOP_NUM;
}

await runBars({
  onBatch(state, batch) {
    const crossUp = new Set<string>();
    const dip = new Set<string>();
    const fresh = new Set<string>();
    for (const bar of batch.bars) {
      const s = pushBar(bar);
      fresh.add(bar.symbol);
      if (bar.v === '0') continue;
      if (s.gap === null || s.now === null) continue;
      if (s.now > 0 && s.gap <= 0) crossUp.add(bar.symbol);
      if (s.now < 0 && s.gap >= 0) s.down = true;
      if (s.now > 0) {
        const cci = cciOf(s.typicals);
        if (cci !== null && cci <= DIP_CCI_AT_MOST) dip.add(bar.symbol);
      }
    }
    if (state.warmingUp) return [];
    const { caps, fees } = state.hello;
    const sells: SellInput[] = [];
    const entries: string[] = [];
    for (const { symbol } of state.hello.symbols) {
      if (!fresh.has(symbol)) continue;
      const held = state.positions.get(symbol);
      if (held !== undefined) {
        if (exits(state, symbol, held, series.get(symbol)!.down)) sells.push({ side: 'sell', symbol, qty: held.qty, max_slippage_bps: MAX_SLIPPAGE_BPS });
      } else if (crossUp.has(symbol) || dip.has(symbol)) {
        entries.push(symbol);
      }
    }

    // Momentum rank: 24-bar return, best first. Only once the history
    // exists; before that every symbol passes the filter.
    const ranked = state.hello.symbols
      .map(({ symbol }) => ({ symbol, closes: series.get(symbol)?.closes ?? [] }))
      .filter((x) => x.closes.length > RANK_LOOKBACK)
      .sort((a, b) => (b.closes[b.closes.length - 1] / b.closes[0]) - (a.closes[a.closes.length - 1] / a.closes[0]))
      .map((x) => x.symbol);
    const rankedOn = ranked.length > 0;
    const allowed = new Set(ranked.slice(0, RANK_TOP));

    const orders: (SellInput | BuyInput)[] = sells.slice(0, caps.max_orders_per_bar);
    const slots = Math.min(state.hello.symbols.length, caps.max_open_positions);
    const stake = state.equity / BigInt(slots);
    const byWeight = (BigInt(caps.max_weight_bps) * state.equity) / BPS;
    let room = slots - state.positions.size;
    let cash = state.cash;
    for (const symbol of entries) {
      if (room <= 0 || orders.length >= caps.max_orders_per_bar) break;
      if (rankedOn && !allowed.has(symbol)) continue;
      let slice = stake < byWeight ? stake : byWeight;
      if (cash < slice) slice = cash;
      if (slice === 0n || slice < BigInt(fees.min_notional_quote_units)) break;
      orders.push({ side: 'buy', symbol, notional_quote_units: slice, max_slippage_bps: MAX_SLIPPAGE_BPS });
      cash -= slice;
      room--;
    }
    return orders;
  },
});
