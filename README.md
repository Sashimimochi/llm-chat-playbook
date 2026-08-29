# LLM Chat Playbook

## Purpose

実装した内容をGitHubのDraft Pull Requestとして公開する。

## Rules

- main/masterへ直接commitしない。
- force pushしない。
- 作業開始時に専用branchを作成する。
- branch名は以下の形式:
  - feature/<short-description>
  - fix/<short-description>
  - refactor/<short-description>
  - chore/<short-description>

## Workflow

1. 現在のbranchを確認する。
2. main/masterへ戻る。
3. originのmain/masterをpullする。
4. 作業branchを作成する。
5. 実装する。
6. テストを実行する。
7. 差分を確認する。
8. commitする。
9. originへpushする。
10. `scripts/create-pr.sh` を実行してDraft PRを作成する。
11. PR URLを報告する。

## Important

GitHubへのPR作成には `gh pr create` を使用する。

Git操作で迷った場合は推測せず、
`git status`、`git branch --show-current`、
`git remote -v` 等で現在状態を確認する。
