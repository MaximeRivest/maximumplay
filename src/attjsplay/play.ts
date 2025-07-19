import { attachments } from 'attachments-js';

const ctx = await attachments(
  'report.pdf',
  'https://example.com/slides.pptx[3-5]',
);

console.log(String(ctx));   // → model‑ready text
console.log(ctx.images);    // → base64 PNGs


import { attachments } from 'attachments-js';
import Anthropic from '@anthropic-ai/sdk';
const anthropic = new Anthropic();

const messageForClaude = await pipe(
    attach('https://example.com/data.csv[limit:10]')
        )(
    load.urlToDataFrame
        )(
    modify.limit
        )(
    present.text
        )(
    adapt.claude, 'What is in this data?'
  )();          // ← unwrap final value

const msg = await anthropic.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: messageForClaude,
});
console.log(msg);