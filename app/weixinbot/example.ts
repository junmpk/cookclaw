/**
 * WeixinBot SDK 使用示例
 *
 * 运行方式：
 *   npx tsx example.ts
 */
import { WeixinBot } from "./src/index.js";
import { fingerprint, safeError } from "./src/utils.js";

async function main() {
  const bot = new WeixinBot({
    // token: "your-bot-token",  // 如果已有 token，直接填入
    stateDir: "~/.weixinbot",
    logLevel: "DEBUG",
    onMessage: async (message, ctx) => {
      console.log("──────────────────────────────────────");
      console.log(`收到消息 from=${fingerprint(ctx.fromUserId)} text_chars=${ctx.text.length}`);
      if (ctx.imagePath) console.log("  图片: [redacted path]");
      if (ctx.voicePath) console.log("  语音: [redacted path]");
      if (ctx.filePath) console.log("  文件: [redacted path]");
      if (ctx.videoPath) console.log("  视频: [redacted path]");

      // 回复文本
      await ctx.replyText("你好，消息已收到。");

      // 发送 typing 状态
      await ctx.sendTyping();
    },
  });

  // ─── 方式一：QR 码登录 ───
  if (!process.env.WEIXIN_BOT_TOKEN) {
    console.log("未设置 WEIXIN_BOT_TOKEN，开始 QR 码登录...");
    const result = await bot.loginWithQRCode({
      onQRCode: (url) => {
        console.log(`\n请扫描二维码登录：\n${url}\n`);
      },
    });
    if (!result.success) {
      console.error("登录失败:", result.message);
      process.exit(1);
    }
    console.log(`登录成功！account=${fingerprint(result.accountId)} token_configured=${Boolean(bot.getToken())}`);
  }

  // ─── 方式二：使用已有 token ───
  // bot.setToken("your-token-here");

  // ─── 主动发消息 ───
  // await bot.sendText("user@im.wechat", "Hello from WeixinBot!");
  // await bot.sendMedia("/path/to/image.png", "user@im.wechat", "图片说明");

  // ─── 启动消息监听 ───
  console.log("开始监听消息...");
  await bot.start();
}

main().catch((err) => {
  console.error(`Fatal error: ${safeError(err)}`);
  process.exit(1);
});
