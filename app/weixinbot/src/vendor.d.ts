declare module "qrcode-terminal" {
  const qr: {
    generate(text: string, opts?: { small?: boolean }, callback?: (code: string) => void): void;
  };
  export default qr;
}

declare module "silk-wasm" {
  export function decode(data: Buffer | Uint8Array, sampleRate: number): Promise<{
    data: Uint8Array;
    duration: number;
  }>;
}
