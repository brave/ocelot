from playwright.sync_api import sync_playwright
import sys
import time
import random
import os
import platform
import multiprocessing
import argparse
import shutil
import gc
from pathlib import Path
from tqdm import tqdm

# Chromium src checkout: same tree `npm run start` uses — Playwright needs the binary, not npm.
# Default: ~/brave-browser/src + out/Component_arm64 (override with BRAVE_SRC / BRAVE_OUT_CONFIG).
_DEFAULT_BRAVE_SRC = Path.home() / "brave-browser" / "src"
DEFAULT_BRAVE_SRC_ROOT = os.environ.get("BRAVE_SRC", "").strip() or str(_DEFAULT_BRAVE_SRC)
DEFAULT_BRAVE_OUT_CONFIG = os.environ.get("BRAVE_OUT_CONFIG", "Component_arm64").strip() or "Component_arm64"
def _default_user_data_dir() -> str:
    env = os.environ.get("BRAVE_USER_DATA_DIR", "").strip()
    if env:
        return env
    home = Path.home()
    if platform.system() == "Darwin":
        return str(
            home
            / "Library/Application Support/BraveSoftware/Brave-Browser-Nightly"
        )
    return str(home / ".config/BraveSoftware/Brave-Browser-Nightly")


DEFAULT_USER_DATA_DIR = _default_user_data_dir()
_OCELOT_AI_CHAT_SERVER_URL = os.environ.get(
    "OCELOT_AI_CHAT_SERVER_URL", "http://127.0.0.1:8000"
).strip()
DEFAULT_BROWSER_ARGS = [
    "--force-device-scale-factor=0.75",
    f"--ai-chat-server-url={_OCELOT_AI_CHAT_SERVER_URL}",
    "--enable-features=AIChatConversationAPIV2",
]


def resolve_brave_dev_executable(src_root: str | Path, out_config: str = "Default") -> Path | None:
    """
    Return path to a locally built Brave binary (the one `npm run start` launches from `src/`).
    src_root: directory that contains `out/` (e.g. brave-browser/src).
    """
    root = Path(src_root).expanduser().resolve()
    out = root / "out" / out_config
    system = platform.system()
    if system == "Darwin":
        candidate = out / "Brave Browser Development.app" / "Contents" / "MacOS" / "Brave Browser Development"
        return candidate if candidate.is_file() else None
    if system == "Linux":
        candidate = out / "brave development"
        return candidate if candidate.is_file() else None
    # Windows: brave.exe next to chrome.dll in out dir
    if system == "Windows":
        candidate = out / "brave development.exe"
        return candidate if candidate.is_file() else None
    return None


def default_browser_executable() -> str | None:
    """BRAVE_DEV_EXECUTABLE if set, else binary resolved from BRAVE_SRC + BRAVE_OUT_CONFIG."""
    env_bin = os.environ.get("BRAVE_DEV_EXECUTABLE", "").strip()
    if env_bin:
        return env_bin
    resolved = resolve_brave_dev_executable(DEFAULT_BRAVE_SRC_ROOT, DEFAULT_BRAVE_OUT_CONFIG)
    return str(resolved) if resolved is not None else None


def check_executable_exists(executable_path):
    """
    Check if the browser executable exists.
    Returns True if it exists, False otherwise.
    """
    if not executable_path:
        return False
    if os.path.isfile(executable_path):
        return True
    # Also check if it's executable by name (in PATH)
    import shutil
    full_path = shutil.which(executable_path)
    return full_path is not None


def dismiss_popups(page, max_wait_seconds=4):
    try:
        page.keyboard.press("Escape")
        time.sleep(0.5)
        cookie_selectors = [
            'button:has-text("Accept")',
            'button:has-text("Accept all")',
            'button:has-text("Accept All")',
            'button:has-text("Consent")',
            'button:has-text("consent")',
            'button:has-text("Consent to all")',
            'button:has-text("Consent to all cookies")',
            'button:has-text("Consent to cookies")',
            'button:has-text("Consent to all tracking")',
            'button:has-text("Consent to all tracking cookies")',
            'button:has-text("Consent To All")',
            'button:has-text("Accept and continue")',
            'button:has-text("Accept and proceed")',
            'button:has-text("Accept and proceed to the site")',
            'button:has-text("Accept and proceed to the website")',
            'button:has-text("Accept and proceed to the web page")',
            'button:has-text("Accept and proceed to the web page")',
            'button:has-text("Allow all")',
            'button:has-text("Allow All")',
            'button:has-text("Agree")',
            'button:has-text("I Agree")',
            'button:has-text("I agree")',
            'button:has-text("Got it")',
            'button:has-text("OK")',
            'button:has-text("Ok")',
            'button:has-text("Close")',
            'button:has-text("Dismiss")',
            '[role="button"]:has-text("Accept")',
            '[role="button"]:has-text("Agree")',
            '[aria-label*="accept" i]',
            '[aria-label*="cookie" i]',
            '[aria-label*="consent" i]',
        ]
        end_time = time.time() + max_wait_seconds
        while time.time() < end_time:
            clicked_any = False
            for selector in cookie_selectors:
                try:
                    if page.locator(selector).first.is_visible(timeout=500):
                        page.locator(selector).first.click()
                        time.sleep(0.5)
                        clicked_any = True
                        break
                except:
                    continue
            if clicked_any:
                break
            time.sleep(0.5)
    except:
        pass


def has_cloudflare_challenge(page):
    """
    Detect if a Cloudflare 'verify you are human' challenge is present.
    Returns True if a challenge is detected, False otherwise.
    """
    try:
        # Wait a bit for Cloudflare challenge to appear
        time.sleep(1)
        
        # Check for Cloudflare challenge indicators - check page content first
        page_text = page.content()
        cloudflare_indicators = [
            "challenges.cloudflare.com",
            "cf-browser-verification",
            "cf-challenge",
            "Just a moment",
            "Checking your browser",
            "Verify you are human",
            "cf-turnstile",
        ]
        
        has_cloudflare = any(indicator in page_text for indicator in cloudflare_indicators)
        
        if not has_cloudflare:
            # Also check for iframes
            try:
                iframe_count = page.locator('iframe').count()
                if iframe_count > 0:
                    # Check if any iframe is from Cloudflare
                    for i in range(iframe_count):
                        try:
                            iframe = page.locator('iframe').nth(i)
                            src = iframe.get_attribute('src') or ''
                            if 'cloudflare' in src.lower() or 'challenges' in src.lower():
                                has_cloudflare = True
                                break
                        except:
                            continue
            except:
                pass
        
        return has_cloudflare
        
    except Exception as e:
        # If we can't check, assume no challenge
        return False


def wait_for_page_ready(page, timeout=10000):
    """
    Wait for a page to be ready by checking multiple indicators.
    More reliable than just waiting for networkidle which can fail on pages
    with continuous network activity.
    """
    try:
        # Wait for document ready state
        page.wait_for_function(
            "document.readyState === 'complete' || document.readyState === 'interactive'",
            timeout=timeout
        )
    except:
        pass  # Page might already be ready


def copy_user_profile_excluding_locks(source_dir, dest_dir):
    """
    Copy user profile directory excluding lock files that can't be copied.
    Lock files (SingletonLock, SingletonCookie, SingletonSocket) are special
    files that shouldn't be copied and will be created by the browser when needed.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # Files to exclude (lock files that can't be copied)
    exclude_files = {"SingletonLock", "SingletonCookie", "SingletonSocket"}
    
    def ignore_function(dir_path, names):
        """Return list of names to ignore during copy"""
        ignored = []
        for name in names:
            if name in exclude_files:
                ignored.append(name)
        return ignored
    
    # Use copytree with ignore function to skip lock files
    try:
        shutil.copytree(source_path, dest_path, ignore=ignore_function, dirs_exist_ok=True)
    except Exception as e:
        # If copytree fails, try manual copying
        print(f"Warning: copytree failed, trying manual copy: {e}")
        dest_path.mkdir(parents=True, exist_ok=True)
        for item in source_path.iterdir():
            if item.name in exclude_files:
                continue
            dest_item = dest_path / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, dest_item, ignore=ignore_function, dirs_exist_ok=True)
                elif item.is_file():
                    shutil.copy2(item, dest_item)
            except (OSError, IOError) as copy_error:
                print(f"Warning: Could not copy {item.name}: {copy_error}")
                continue


def process_url(args_tuple):
    """
    Process a single URL by launching a browser directly with Playwright.
    This function runs in a separate process with its own Playwright instance.
    """
    url, worker_id, min_response_delay, max_response_delay, browser_executable, browser_args, user_data_dir, headless = args_tuple
    
    try:
        # Each process needs its own Playwright instance
        with sync_playwright() as playwright:
            # Launch browser directly with Playwright with user profile using persistent context
            worker_user_data_dir = f"{user_data_dir}_worker_{worker_id}"
            
            # Clean up any lock files left from previous browser sessions
            worker_dir_path = Path(worker_user_data_dir)
            lock_files = [
                worker_dir_path / "SingletonLock",
                worker_dir_path / "SingletonSocket",
                worker_dir_path / "SingletonCookie",
            ]
            for lock_file in lock_files:
                if lock_file.exists():
                    try:
                        lock_file.unlink()
                    except Exception as e:
                        print(f"[Worker {worker_id}] Warning: Could not remove lock file {lock_file.name}: {e}")
            
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=worker_user_data_dir,
                executable_path=browser_executable,
                args=browser_args,
                headless=headless,
                accept_downloads=False
            )
            
            # Block downloads by canceling any download attempts
            def cancel_download(download):
                print(f"[Worker {worker_id}] Blocked download attempt: {download.url}")
                download.cancel()
            context.on("download", cancel_download)
            
            url_page = None
            leo_page = None
            
            try:
                # Step 1: Navigate to the URL first
                url_page = context.new_page()
                url_page.on("download", cancel_download)
                url_page.set_default_navigation_timeout(20000)
                url_page.set_default_timeout(20000)
                
                # Navigate with retry logic for HTTP/2 protocol errors
                # ERR_NET_HTTP2_PROTOCOL errors can occur due to server-side HTTP/2 issues
                max_navigation_retries = 3
                navigation_success = False
                
                for attempt in range(max_navigation_retries):
                    try:
                        # Use "load" instead of "networkidle" - more reliable as it waits for the load event
                        # networkidle can fail on pages with continuous network activity (analytics, ads, etc.)
                        url_page.goto(url, wait_until="load", timeout=10000)
                        navigation_success = True
                        break
                    except Exception as e:
                        error_str = str(e)
                        error_upper = error_str.upper()
                        # Check if it's an HTTP/2 protocol error
                        if (
                            "ERR_NET_HTTP2_PROTOCOL" in error_str
                            or "ERR_HTTP2" in error_upper
                            or "ERR_NET2" in error_upper
                            or "HTTP2_PROTOCOL" in error_upper
                            or "HTTP2" in error_upper
                        ):
                            if attempt < max_navigation_retries - 1:
                                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                                print(f"[Worker {worker_id}] HTTP/2 protocol error (attempt {attempt + 1}/{max_navigation_retries}), retrying in {wait_time}s...")
                                time.sleep(wait_time)
                                continue
                            else:
                                print(f"[Worker {worker_id}] HTTP/2 protocol error after {max_navigation_retries} attempts: {e}")
                                # Try fallback methods
                                try:
                                    url_page.goto(url, wait_until="domcontentloaded", timeout=10000)
                                    wait_for_page_ready(url_page, timeout=5000)
                                    navigation_success = True
                                    break
                                except Exception as e2:
                                    print(f"[Worker {worker_id}] Fallback navigation also failed: {e2}")
                                    raise
                        else:
                            # For other errors, try fallback methods
                            if attempt < max_navigation_retries - 1:
                                print(f"[Worker {worker_id}] Navigation error (attempt {attempt + 1}/{max_navigation_retries}), trying fallback: {e}")
                                try:
                                    url_page.goto(url, wait_until="domcontentloaded", timeout=10000)
                                    wait_for_page_ready(url_page, timeout=5000)
                                    navigation_success = True
                                    break
                                except:
                                    time.sleep(1)
                                    continue
                            else:
                                # Last resort: just navigate and wait
                                print(f"[Worker {worker_id}] Navigation fallback: {e}")
                                url_page.goto(url, timeout=10000)
                                wait_for_page_ready(url_page, timeout=5000)
                                navigation_success = True
                                break
                
                if not navigation_success:
                    raise Exception(f"Failed to navigate to {url} after {max_navigation_retries} attempts")
                
                # Check for Cloudflare challenge - if present, skip this URL
                if has_cloudflare_challenge(url_page):
                    print(f"[Worker {worker_id}] Cloudflare challenge detected for {url}, skipping...")
                    return {"url": url, "status": "ERROR: Cloudflare challenge detected, skipped"}
                
                dismiss_popups(url_page)
                time.sleep(1)
                
                # Step 2: Retrieve the loaded URL and page title from that page (after redirects)
                # Wait for title to be available (indicates page is loaded)
                try:
                    url_page.wait_for_function("document.title !== ''", timeout=5000)
                except:
                    pass  # Title might already be available
                
                loaded_url = url_page.url
                page_title = url_page.title()
                
                # Step 3: Open a Leo tab
                leo_page = context.new_page()
                leo_page.on("download", cancel_download)
                
                # Navigate to brave://leo-ai/
                # For internal browser pages, domcontentloaded is usually sufficient
                try:
                    leo_page.goto("brave://leo-ai/", wait_until="domcontentloaded", timeout=30000)
                except Exception as e:
                    print(f"[Worker {worker_id}] Standard navigation failed: {e}, trying alternative methods...")
                    try:
                        leo_page.evaluate('window.location.href = "brave://leo-ai/"')
                        # Wait for navigation to complete
                        leo_page.wait_for_load_state("domcontentloaded", timeout=10000)
                        time.sleep(1)
                    except Exception as e2:
                        print(f"[Worker {worker_id}] JavaScript navigation also failed: {e2}")
                        try:
                            leo_page.bring_to_front()
                            time.sleep(0.5)
                            leo_page.keyboard.press("Control+l")
                            time.sleep(1)
                            leo_page.keyboard.type("brave://leo-ai/")
                            time.sleep(0.5)
                            leo_page.keyboard.press("Enter")
                            # Wait for navigation
                            try:
                                leo_page.wait_for_load_state("domcontentloaded", timeout=10000)
                            except:
                                pass
                            time.sleep(1)
                        except Exception as e3:
                            print(f"[Worker {worker_id}] All navigation methods failed: {e3}")
                
                # Wait for page to be ready - check if it's actually loaded
                time.sleep(1)
                dismiss_popups(leo_page)
                time.sleep(1)
                
                # Find the text input
                text_input = None
                selectors = [
                    '[data-placeholder*="How can I help"]',
                    'input[data-placeholder*="How can I help"]',
                    'textarea[data-placeholder*="How can I help"]',
                    '[data-placeholder="How can I help you today?"]',
                    'input[data-placeholder="How can I help you today?"]',
                    'textarea[data-placeholder="How can I help you today?"]',
                ]
                
                for selector in selectors:
                    try:
                        locator = leo_page.locator(selector).first
                        locator.wait_for(state="visible", timeout=5000)
                        if locator.is_visible():
                            text_input = locator
                            break
                    except Exception as e:
                        continue
                
                if not text_input:
                    raise Exception("Could not find text input field")
                
                # Step 4: Type "@" followed by the tab name (page title)
                text_input.click()
                time.sleep(0.5)
                text_input.fill(f"@{page_title}")
                time.sleep(0.5)
                text_input.press("Enter")
                time.sleep(0.5)
                
                # Click the attachment icon button
                attachment_button = None
                try:
                    # Look for leo-icon with name="attachment"
                    attachment_button = leo_page.locator('leo-icon[name="attachment"]')
                    attachment_button.wait_for(state="visible", timeout=5000)
                    if not attachment_button.is_visible():
                        raise Exception("Attachment icon not visible")
                except Exception as e:
                    # Try alternative selectors
                    try:
                        attachment_button = leo_page.locator('leo-icon[name="attachment"]').locator('..')
                        attachment_button.wait_for(state="visible", timeout=5000)
                    except:
                        raise Exception(f"Could not find attachment icon: {e}")
                
                if not attachment_button:
                    raise Exception("Could not find attachment icon button")
                
                attachment_button.click()
                time.sleep(0.5)
                
                # Click the Screenshot icon
                screenshot_icon = None
                try:
                    # Look for leo-icon with name="screenshot"
                    screenshot_icon = leo_page.locator('leo-icon[name="screenshot"]')
                    screenshot_icon.wait_for(state="visible", timeout=5000)
                    if not screenshot_icon.is_visible():
                        raise Exception("Screenshot icon not visible")
                except Exception as e:
                    # Try alternative selectors
                    try:
                        screenshot_icon = leo_page.locator('leo-icon[name="screenshot"]').locator('..')
                        screenshot_icon.wait_for(state="visible", timeout=5000)
                    except:
                        raise Exception(f"Could not find screenshot icon: {e}")
                
                if not screenshot_icon:
                    raise Exception("Could not find screenshot icon")
                
                screenshot_icon.click()
                time.sleep(1)  # Wait for screenshot to be taken and editor to appear
                
                # Find the span editor with the specific placeholder
                screenshot_editor = None
                try:
                    screenshot_editor = leo_page.locator('span[data-editor="true"][data-placeholder="Ask a question about the content of this page"]')
                    screenshot_editor.wait_for(state="visible", timeout=10000)
                    if not screenshot_editor.is_visible():
                        raise Exception("Screenshot editor not visible")
                except Exception as e:
                    raise Exception(f"Could not find screenshot editor: {e}")
                
                # Type "Summarise" into the contenteditable span
                screenshot_editor.click()
                time.sleep(0.5)
                screenshot_editor.fill("Summarise")
                time.sleep(0.5)
                screenshot_editor.press("Enter")
                
                # Wait for response to complete
                wait_time = random.uniform(min_response_delay, max_response_delay)
                time.sleep(wait_time)
                
                result = {"url": url, "loaded_url": loaded_url, "status": "success"}
                
            finally:
                # CRITICAL: Close all tabs BEFORE the context/window closes
                # This must happen inside the with block while Playwright is still active
                if url_page and not url_page.is_closed():
                    try:
                        url_page.close()
                    except Exception:
                        pass
                
                if leo_page and not leo_page.is_closed():
                    try:
                        leo_page.close()
                    except Exception:
                        pass
                
                # Close any remaining pages
                if context:
                    try:
                        for page in list(context.pages):
                            if not page.is_closed():
                                try:
                                    page.close()
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    
                    # Wait for pages to close
                    time.sleep(0.2)
                    
                    # Close context (this closes the window)
                    try:
                        context.close()
                    except Exception:
                        pass
            
            return result
            
    except Exception as e:
        import traceback
        error_msg = f"{url}: {str(e)}\n{traceback.format_exc()}\n"
        script_dir = Path(__file__).parent
        errors_file = script_dir / "errors.txt"
        with open(errors_file, "a") as f:
            f.write(error_msg)
        print(f"[Worker {worker_id}] ERROR processing {url}: {str(e)}")
        return {"url": url, "status": f"ERROR: {str(e)}"}
    finally:
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Summarize web pages using Leo AI")
    parser.add_argument(
        "--urls-file",
        type=str,
        default="urls.txt",
        help="Path to file containing URLs (one per line)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel browser instances (default: 2). Reduce this if running out of memory, especially in headless mode",
    )
    parser.add_argument(
        "--min-response-delay",
        type=int,
        default=20,
        help="Delay in seconds after clicking summarize button (default: 20)"
    )
    parser.add_argument(
        "--max-response-delay",
        type=int,
        default=30,
        help="Maximum delay in seconds after clicking summarize button (default: 30)"
    )
    parser.add_argument(
        "--brave-src",
        type=str,
        default=None,
        metavar="SRC_DIR",
        help="Chromium src dir that contains out/ (default without this flag: ~/brave-browser/src). "
        "Overrides BRAVE_SRC. Ignored if --browser-executable is set.",
    )
    parser.add_argument(
        "--brave-out-config",
        type=str,
        default=DEFAULT_BRAVE_OUT_CONFIG,
        metavar="CONFIG",
        help="GN output folder under out/ (default: Component_arm64, or BRAVE_OUT_CONFIG). E.g. Default, Debug.",
    )
    parser.add_argument(
        "--browser-executable",
        type=str,
        default=None,
        help="Full path to the Brave/Chromium binary. If omitted: BRAVE_DEV_EXECUTABLE, or "
        "resolved from ~/brave-browser/src + out/Component_arm64 (override with BRAVE_SRC / BRAVE_OUT_CONFIG).",
    )
    parser.add_argument(
        "--browser-args",
        type=str,
        nargs="+",
        default=DEFAULT_BROWSER_ARGS,
        help=f"Additional browser arguments (default: {DEFAULT_BROWSER_ARGS})"
    )
    parser.add_argument(
        "--user-data-dir",
        type=str,
        default=DEFAULT_USER_DATA_DIR,
        help=(
            "Optional template profile copied into each worker dir before launch; if missing, workers "
            f"use empty profiles. Default: {DEFAULT_USER_DATA_DIR} or BRAVE_USER_DATA_DIR"
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run browser without a window (default: off — browser is visible). Lowers resource usage; each worker uses significant memory — reduce --num-workers if needed",
    )
    
    args = parser.parse_args()

    if args.browser_executable:
        browser_executable = args.browser_executable
    elif args.brave_src:
        resolved = resolve_brave_dev_executable(args.brave_src, args.brave_out_config)
        if resolved is None:
            out = Path(args.brave_src).expanduser().resolve() / "out" / args.brave_out_config
            print(
                f"Error: No Brave binary found under {out} "
                f"(Darwin: Brave Browser.app/.../Brave Browser, Linux: brave, Windows: brave.exe). "
                f"Build first (npm run build / autoninja) or fix --brave-out-config."
            )
            return
        browser_executable = str(resolved)
    else:
        browser_executable = default_browser_executable()
        if browser_executable is None:
            root = Path(DEFAULT_BRAVE_SRC_ROOT).expanduser().resolve()
            out = root / "out" / DEFAULT_BRAVE_OUT_CONFIG
            print(
                "Error: No Brave binary resolved.\n"
                f"  Expected under: {out}\n"
                "Set BRAVE_DEV_EXECUTABLE to the binary, or BRAVE_SRC / BRAVE_OUT_CONFIG, "
                "or pass --browser-executable / --brave-src."
            )
            return

    if not check_executable_exists(browser_executable):
        print(
            "Error: Browser executable not found:\n"
            f"  {browser_executable}\n"
            "Use --browser-executable PATH, or --brave-src PATH_TO_SRC (with out/ built), "
            "or set BRAVE_DEV_EXECUTABLE / BRAVE_SRC."
        )
        return

    # Get script directory
    script_dir = Path(__file__).parent
    urls_file = script_dir / args.urls_file
    
    # Load URLs
    try:
        with open(urls_file, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: {urls_file} not found")
        return
    
    try:
        # Prepare browser args
        browser_args = args.browser_args.copy()
        
        # Add network stability arguments to handle HTTP/2 protocol errors
        # Note: ERR_NET_HTTP2_PROTOCOL errors can occur due to server-side HTTP/2 issues
        # These flags help with network stability and retries
        network_stability_args = [
            "--disable-quic",  # Disable QUIC to use more stable HTTP protocols
            "--aggressive-cache-discard",  # Better cache handling
        ]
        browser_args.extend(network_stability_args)
        
        # Add memory-saving arguments in headless mode
        if args.headless:
            memory_saving_args = [
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-software-rasterizer",
                "--disable-background-networking",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
            ]
            browser_args.extend(memory_saving_args)
        
        print(f"Each worker will launch: {browser_executable}")
        print(f"Browser arguments: {' '.join(browser_args)}")
        print(f"Loaded {len(urls)} URLs")
        print(f"Using {args.num_workers} parallel workers")
        print(f"Response delay: Random {args.min_response_delay}-{args.max_response_delay} seconds after clicking button")
        
        # Pre-copy user profiles for all workers to avoid conflicts
        print("Preparing worker profile directories...")
        source_dir_path = Path(args.user_data_dir)
        
        # Clean up any existing worker directories first
        print("Cleaning up existing worker directories...")
        for worker_id in range(args.num_workers):
            worker_user_data_dir = f"{args.user_data_dir}_worker_{worker_id}"
            worker_dir_path = Path(worker_user_data_dir)
            if worker_dir_path.exists():
                try:
                    shutil.rmtree(worker_dir_path)
                    print(f"Deleted worker {worker_id} directory")
                except Exception as e:
                    print(f"Warning: Could not delete worker {worker_id} directory: {e}")
        
        # Copy fresh profiles for all workers
        for worker_id in range(args.num_workers):
            worker_user_data_dir = f"{args.user_data_dir}_worker_{worker_id}"
            worker_dir_path = Path(worker_user_data_dir)
            
            print(f"Copying user profile for worker {worker_id}...")
            if source_dir_path.exists():
                copy_user_profile_excluding_locks(source_dir_path, worker_dir_path)
                print(f"Worker {worker_id} profile ready")
            else:
                print(f"Warning: Source user data directory does not exist: {args.user_data_dir}")
        print("All worker profiles prepared")
        
        # Prepare arguments for worker processes
        worker_args = [
            (url, worker_id % args.num_workers, args.min_response_delay, args.max_response_delay,
             browser_executable, browser_args, args.user_data_dir, args.headless)
            for worker_id, url in enumerate(urls)
        ]
        
        # Process URLs with multiprocessing
        results = []
        
        print(f"Processing URLs...")
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            # Use imap_unordered for better performance and to process results as they come in
            with tqdm(total=len(urls), desc="Processing URLs") as pbar:
                for result in pool.imap_unordered(process_url, worker_args):
                    results.append(result)
                    pbar.update(1)
        
        print(f"\nDone! Processed {len(results)} URLs")
        success_n = sum(1 for r in results if r["status"] == "success")
        print(f"Success: {success_n}")
        print(f"Errors: {sum(1 for r in results if 'ERROR' in r['status'])}")
        if len(urls) > 0 and success_n == 0:
            sys.exit(1)
        
        # Clean up worker directories after processing
        print("\nCleaning up worker directories...")
        for worker_id in range(args.num_workers):
            worker_user_data_dir = f"{args.user_data_dir}_worker_{worker_id}"
            worker_dir_path = Path(worker_user_data_dir)
            if worker_dir_path.exists():
                try:
                    shutil.rmtree(worker_dir_path)
                    print(f"Deleted worker {worker_id} directory")
                except Exception as e:
                    print(f"Warning: Could not delete worker {worker_id} directory: {e}")
        print("Cleanup complete")
    
    except Exception as e:
        print(f"Error: {e}")
        # Clean up worker directories even on error
        print("\nCleaning up worker directories after error...")
        for worker_id in range(args.num_workers):
            worker_user_data_dir = f"{args.user_data_dir}_worker_{worker_id}"
            worker_dir_path = Path(worker_user_data_dir)
            if worker_dir_path.exists():
                try:
                    shutil.rmtree(worker_dir_path)
                except:
                    pass
        raise

if __name__ == "__main__":
    main()
