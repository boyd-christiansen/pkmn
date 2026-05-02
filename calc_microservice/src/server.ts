import express from 'express';
import { runCalc } from './calc.js';
import type { CalcRequest } from './types.js';

const app = express();
app.use(express.json({ limit: '256kb' }));

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

const port = Number(process.env.PORT ?? 3000);
app.listen(port, () => {
  console.log(`calc-microservice listening on http://localhost:${port}`);
});
