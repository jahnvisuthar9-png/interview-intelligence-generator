# Put the app online for free (Render)

You only need a browser. Anyone you share the link with can use the app.

## 1. Put the code on GitHub
1. Create a free account at github.com.
2. Click **+** (top right) > **New repository**. Name it `interview-intelligence-generator`, choose **Private**, click **Create repository**.
3. On the new repo page click **uploading an existing file**.
4. Unzip `interview-intelligence-generator.zip` on your computer, open the folder, select **everything inside it** (including the `pipeline`, `templates`, `fonts` and `tests` folders), and drag it into the page.
5. Click **Commit changes**.

## 2. Deploy on Render
1. Create a free account at render.com and sign in with GitHub.
2. Click **New** > **Blueprint**, pick your repository. Render finds `render.yaml` automatically.
3. When asked for `OPENAI_API_KEY`, paste your key. Click **Apply**.
4. Wait about 3-5 minutes for the first build. Your link appears at the top, like
   `https://interview-intelligence-generator.onrender.com`. Share that link.

## Things to know
- **First visit after a quiet period is slow.** The free plan sleeps after 15 minutes without visitors; the next visit takes about a minute to wake it. After that it's normal speed.
- **Your OpenAI credit pays for every generation.** Anyone with the link can use it. Set a monthly spending limit in your OpenAI account (Settings > Limits).
- **Generated files are temporary.** They're deleted when the app sleeps or restarts, so download the PDF/ZIP right away.
- **Updating the app:** upload changed files to the GitHub repo; Render redeploys automatically.
- **Changing the key or model:** Render dashboard > your service > Environment.
