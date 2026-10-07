// Development-only model fixture. It tests wiring, not semantic relevance.
import sharp from 'sharp';

const server = Bun.serve({
  hostname: '127.0.0.1',
  port: Number(process.env.MOCK_PORT || 4097),
  async fetch(req) {
    const body = (await req.json()) as any;
    const parts = body.input?.[0]?.content || [];
    const vector = new Float32Array(768);
    const picture = parts.find((p: any) => p.type === 'image_url');
    if (picture) {
      const bytes = Buffer.from(picture.image_url.url.split(',')[1], 'base64');
      const stats = await sharp(bytes).stats();
      stats.channels.forEach((c, i) => (vector[i] = c.mean / 255));
      vector[3] = stats.entropy / 10;
    } else {
      vector[0] = 0.5;
      vector[1] = 0.3;
      vector[2] = 0.2;
    }
    return Response.json({ data: [{ embedding: Array.from(vector) }] });
  },
});
console.log(`Mock embedding server on ${server.port}`);
