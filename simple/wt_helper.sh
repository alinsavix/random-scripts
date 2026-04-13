# A script for making it a bit easier to use worktrees in git. Must be
# sourced, not run directly (otherwise it couldn't "cd" to worktrees for
# you). Worktrees will be created under <repo>/.worktrees/, which will also
# be locally excluded from git tracking.

# Throw an error if executed directly
if [ -n "$BASH_VERSION" ] && [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'ERROR: wt_helper.sh must be sourced, not executed directly.\n' >&2
    printf '  Use: source %s\n' "$0" >&2
    exit 1
elif [ -n "$ZSH_VERSION" ] && [[ "$ZSH_EVAL_CONTEXT" != *:file* ]]; then
    printf 'ERROR: wt_helper.sh must be sourced, not executed directly.\n' >&2
    printf '  Use: source %s\n' "$0" >&2
    exit 1
fi

# exclude .worktrees/ from git tracking, without requiring any changes to
# the actual repo (e.g. no .gitignore changes)
_wt_ensure_excluded() {
    local toplevel="$1" wtroot="$2"
    if [[ "$wtroot" == "$toplevel"/* ]]; then
        local wtroot_relative="${wtroot#$toplevel/}"
        local exclude_file="$toplevel/.git/info/exclude"
        mkdir -p "$(dirname "$exclude_file")"
        if [ -f "$exclude_file" ]; then
            grep -qF "$wtroot_relative/" "$exclude_file" 2>/dev/null \
                || printf '\n%s/\n' "$wtroot_relative" >> "$exclude_file"
        else
            printf '%s/\n' "$wtroot_relative" > "$exclude_file"
        fi
    fi
}

# make a new worktree (and cd into it)
_wt_create_and_cd() {
    local toplevel="$1" wtroot="$2" branch_name="$3" create_new="$4"

    mkdir -p "$wtroot" || return 1
    _wt_ensure_excluded "$toplevel" "$wtroot"

    # Extract basename if branch has slashes (e.g., u/user/branch -> branch)
    local dirname="${branch_name##*/}"
    local target="$wtroot/$dirname"

    if [ -e "$target" ]; then
        echo "wt: target directory already exists: $target" >&2
        echo "wt: branch basename '$dirname' collides — rename the branch or remove the existing directory" >&2
        return 1
    fi

    if [ "$create_new" = "true" ]; then
        git worktree add -b "$branch_name" "$target" || return 1
    else
        git worktree add "$target" "$branch_name" || return 1
    fi

    cd "$target"
}

_wt_list_branches() {
    git worktree list --porcelain \
        | awk '
            $1=="worktree"{path=$2}
            $1=="branch"{branch=$2; sub(/^refs\/heads\//,"",branch); print path "\t" branch}
        '
}

# find tree by exact branch
_wt_find_by_branch() {
    _wt_list_branches | awk -v q="$1" '$2==q{print $1; exit}'
}

# Find tree by substring (branch or path)
_wt_find_by_query() {
    _wt_list_branches | awk -v q="$1" 'index($0,q){print $1; exit}'
}


wt() {
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "wt: not inside a git repo" >&2
        return 1
    fi

    local toplevel wtroot cmd arg target

    # Get the main repo root (not the current worktree root)
    local gitcommon
    gitcommon="$(git rev-parse --git-common-dir 2>/dev/null)" || return 1
    # Strip /.git suffix to get repo root
    toplevel="${gitcommon%/.git}"
    # If gitcommon didn't end in /.git, it might be a bare .git dir, use show-toplevel
    if [ "$toplevel" = "$gitcommon" ]; then
        toplevel="$(git rev-parse --show-toplevel 2>/dev/null)" || return 1
    fi

    wtroot="$toplevel/.worktrees"

    cmd="$1"
    arg="$2"

    case "$cmd" in
        -h|--help)
        cat <<EOF
Usage: wt [OPTIONS] [name]

Options:
  -h, --help                Show this help message
  -l, --list                List all worktrees
  -b, --branch <name>       Create new branch and worktree at .worktrees/<name>
  -c, --create <name>       Create worktree from existing branch at .worktrees/<name>
  -r, --remove [--force] <name>...
                            Remove one or more worktrees by directory, branch, or path
                            Use --force or -f to force removal (dirty/locked worktrees)
  -p, --prune               Prune stale worktrees (directory no longer exists)
Without options:
  wt                   Fuzzy-pick worktree and cd (requires fzf)
  wt <name>            cd to .worktrees/<name> or match branch/path

Assumes worktrees are under: <repo>/.worktrees/
EOF
        return 0
        ;;

        "")
            if command -v fzf >/dev/null 2>&1; then
                target="$(git worktree list --porcelain \
                    | awk '$1=="worktree"{print $2}' \
                    | fzf --height 40% --reverse)"
                [ -n "$target" ] && cd "$target"
                return $?
            fi
            git worktree list
            echo "Tip: install fzf for interactive selection." >&2
            return 0
            ;;

        -l|--list)
            git worktree list
            return 0
            ;;

        -b|--branch)
            if [ -z "$arg" ]; then
                echo "Usage: wt -b <new-branch>" >&2
                return 2
            fi
            _wt_create_and_cd "$toplevel" "$wtroot" "$arg" true
            return $?
            ;;

        -c|--create)
            if [ -z "$arg" ]; then
                echo "Usage: wt -c <existing-branch>" >&2
                return 2
            fi
            _wt_create_and_cd "$toplevel" "$wtroot" "$arg" false
            return $?
            ;;

        -r|--remove)
            # FIXME: perhaps a bit too much directly in the case statement
            if [ -z "$arg" ]; then
                echo "Usage: wt -r [-f|--force] <name|path> [name|path...]" >&2
                return 2
            fi

            shift  # skip the -r/--remove flag
            local exit_code=0 force_flag=""

            local -a worktrees=()
            for item in "$@"; do
                case "$item" in
                    -f|--force) force_flag="--force" ;;
                    *) worktrees+=("$item") ;;
                esac
            done

            # find the worktree we're trying to reference. Look under
            # <repo>/.worktrees, then see if it's a path relative to cwd,
            # and finally try to match a branch name.
            for worktree_name in "${worktrees[@]}"; do
                target=""
                if [ -d "$wtroot/$worktree_name" ]; then
                    target="$wtroot/$worktree_name"
                elif [ -d "$worktree_name" ]; then
                    target="$worktree_name"
                else
                    target="$(_wt_find_by_branch "$worktree_name")"
                fi

                if [ -z "$target" ]; then
                    echo "wt: no worktree found matching '$worktree_name'" >&2
                    exit_code=1
                    continue
                fi

                # If we're currently inside the worktree being removed, first
                # move us back to the main worktree so that we don't end up
                # stuck in limbo.
                local current_wt abs_target
                current_wt="$(git rev-parse --show-toplevel 2>/dev/null)"
                abs_target="$(cd "$target" 2>/dev/null && pwd)"
                if [ -n "$current_wt" ] && [ "$current_wt" = "$abs_target" ]; then
                    local main_wt
                    main_wt="$(git worktree list --porcelain | awk '$1=="worktree"{print $2; exit}')"
                    if [ -z "$main_wt" ] || [ "$main_wt" = "$abs_target" ]; then
                        echo "wt: cannot escape from '$worktree_name' before removing it" >&2
                        exit_code=1
                        continue
                    fi
                    cd "$main_wt" || { exit_code=$?; continue; }
                fi

                git worktree remove ${force_flag:+"$force_flag"} "$target" || exit_code=$?
            done
            return $exit_code
            ;;

        -p|--prune)
            git worktree prune
            return $?
            ;;
    esac

    # handle "wt <name>"
    # try .worktrees/<name>
    if [ -d "$wtroot/$cmd" ]; then
        cd "$wtroot/$cmd"
        return $?
    fi

    # try branch name or substring match
    target="$(_wt_find_by_query "$cmd")"

    # pick whichever of main/master exists
    if [ -z "$target" ]; then
        local alternate=""
        if [ "$cmd" = "main" ]; then
            alternate="master"
        elif [ "$cmd" = "master" ]; then
            alternate="main"
        fi

        if [ -n "$alternate" ]; then
            target="$(_wt_find_by_query "$alternate")"
        fi
    fi

    if [ -z "$target" ]; then
        echo "wt: no worktree matching '$cmd' (looked under $wtroot and git worktrees)" >&2
        return 1
    fi

    cd "$target"
}
