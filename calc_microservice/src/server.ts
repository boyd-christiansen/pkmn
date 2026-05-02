import express from 'express';
import { runCalc } from './calc.js';
import { lookupMove } from './dex.js';
import { parseLog } from './parse_log.js';
import type { CalcRequest } from './types.js';

const app = express();
app.use(express.json({ limit: '4mb' }));

app.get('/health', (_req, res) => {
  res.json({ status: 'ok' });
});

app.post('/calc', (req, res) => {
  try {
    const body = req.body as CalcRequest;
    if (!body || typeof body !== 'object') {
      return res.status(400).json({ error: 'Body must be a JSON object' });
    }
    const result = runCalc(body);
    return res.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return res.status(400).json({ error: message });
  }
});

app.get('/dex/move/:name', (req, res) => {
  const move = lookupMove(req.params.name);
  if (!move) {
    return res.status(404).json({ error: `move not found: ${req.params.name}` });
  }
  return res.json(move);
});

app.post('/parse_log', (req, res) => {
  try {
    const body = req.body as { log?: unknown };
    if (!body || typeof body !== 'object' || typeof body.log !== 'string') {
      return res
        .status(400)
        .json({ error: 'Body must be a JSON object with a string field "log"' });
    }
    const snapshots = parseLog(body.log);
    return res.json({ snapshots });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return res.status(400).json({ error: message });
  }
});

const port = Number(process.env.PORT ?? 3000);
app.listen(port, () => {
  console.log(`calc-microservice listening on http://localhost:${port}`);
});
