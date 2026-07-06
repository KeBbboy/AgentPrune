# 项目约定（Project Conventions）

本文件记录所有协作者（包括 AI 助手）必须遵守的项目约定。
**每次提交 commit、推送 push 到远程之前，必须先阅读本文件。**

## 1. Git 提交与推送规范

### 1.1 Commit message 必须使用中文描述

- ✅ **正确**：commit message 标题和正文都使用中文
- ❌ **错误**：使用英文或中英文混杂

**示例**：

```bash
# ✅ 正确
git commit -m "feat: 实现智能体之间传递完整的 reasoning 过程"

# ✅ 正确（带正文）
git commit -m "$(cat <<'EOF'
feat: 统一 thinking 参数配置入口

- experiments/run_aime.py、run_swe_bench_lite.py：新增 --llm_enable_thinking
  与 --llm_thinking_budget 命令行参数
- 其余 run_*.py：精简 --llm_thinking_budget 帮助文本
EOF
)"

# ❌ 错误
git commit -m "feat: unify thinking config"
```

### 1.2 禁止添加 Co-Authored-By 等 AI 共同作者 trailer

- AI 助手（Claude / 其它 LLM）参与生成的 commit **不得**在 message 末尾添加
  `Co-Authored-By: ... <noreply@anthropic.com>` 这类 trailer，否则会在 GitHub
  贡献者统计里把 AI 列为共同作者。
- 也不得添加其它形式的 `Signed-off-by:` / `Co-authored-by:` AI 占位 trailer。
- 仅在用户明确要求并指明 trailer 文本时才可以附加。

**示例**：

```bash
# ❌ 错误
git commit -m "$(cat <<'EOF'
feat: xxx

- 改了 A、B、C

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

# ✅ 正确
git commit -m "$(cat <<'EOF'
feat: xxx

- 改了 A、B、C
EOF
)"
```

### 1.3 推送流程

1. 检查当前分支：`git branch --show-current`
2. 确认改动已暂存：`git status`
3. 提交：`git commit -m "中文描述"`
4. 推送到远程：`git push origin <当前分支名>`

## 2. 代码风格约定

- 所有注释和文档优先使用中文
- 函数 docstring 优先中文
- print 输出可以使用中英文

## 3. 修改原则

- 不主动创建新文件（除非必要）
- 优先编辑已有文件
- 不主动创建 README.md 等文档（除非用户明确要求）

## 4. AI 助手必读

每次开始 commit、push 操作前，请先：

1. 读取本文件 `CONVENTIONS.md`（项目根目录）
2. 严格遵守上面的所有约定
3. 特别确认：commit message 使用中文
