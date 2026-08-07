.PHONY: help context validate validate-strict new-task

help:
	@echo "make context          显示 AI Coding 必读入口"
	@echo "make validate         检查结构、链接、Python 语法和课件哈希"
	@echo "make validate-strict  额外核对固定上游源码哈希"
	@echo "make new-task NAME=my-op [TITLE='My Operator']"

context:
	@echo "1. AGENTS.md"
	@echo "2. docs/AI_CONTEXT.md"
	@echo "3. docs/KNOWLEDGE_INDEX.md"
	@echo "4. docs/AI_TASK_RECIPES.md"
	@echo "5. docs/TASK_CATALOG.md"
	@echo "6. docs/OPERATOR_DEVELOPMENT_PLAYBOOK.md"
	@echo "7. workspaces/<task>/TASK.md"

validate:
	python3 scripts/validate_repository.py

validate-strict:
	python3 scripts/validate_repository.py --strict

new-task:
	@test -n "$(NAME)" || (echo "用法: make new-task NAME=my-op"; exit 2)
	python3 scripts/new_operator_task.py "$(NAME)" $(if $(TITLE),--title "$(TITLE)",)
