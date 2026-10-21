# CLAUDE.md

## Git 提交约定（必须遵守）

**每次在 Kaggle 上触发运行（`kaggle kernels push`）之前，都要先单独 git commit 一次。**

- 一次 push 对应一个 commit；同时 push 多个 kernel 时，按 kernel 拆成多个 commit（各自只 add 该 kernel 的 `src/*.py`、`.ipynb`、`kernel-metadata.json`）
- commit message：根据 `git diff --cached` 写一行英文，conventional commit 前缀（`feat` / `fix` / `refac` / `doc` / `chore` 等），尽量简短，结尾不加句号
- 固定流程：改 `src/*.py` → `src/make_notebook.py` 重新生成 ipynb → `git add` → 生成 message → `git commit` → `kaggle kernels push`
- 向 LB 提交 `submission.csv` 时，`-m` 描述里带上产出它的 commit hash，保证榜上分数可追溯

## 本地环境（macOS）

- 虚拟环境由 uv 创建：`uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt`
- 命令用 `.venv/bin/python`、`.venv/bin/kaggle`（README 里的 `.\.venv\Scripts\...` 是 Windows 写法）
- Kaggle 凭据在 `~/.kaggle/access_token`
- LightGBM 本地运行缺 `libomp`，借用 scikit-learn 自带的：
  `export DYLD_LIBRARY_PATH=$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs`
- 本机下载 Kaggle 数据很慢（约 30–80 kB/s），大文件放后台下载
