"""Run genuine two-model inference, then optionally verify the browser score.

python scripts/smoke_real.py recording.wav --url http://127.0.0.1:8000
Add --browser when Playwright/Chromium are installed. No predictions are mocked.
"""
import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--browser', action='store_true')
    args = parser.parse_args()
    with httpx.Client(base_url=args.url, timeout=180) as client:
        deadline = time.monotonic() + 30
        while True:
            try:
                if client.get('/api/health').status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('API did not become healthy within 30 seconds')
            time.sleep(.5)
        started = time.monotonic()
        with args.audio.open('rb') as handle:
            response = client.post('/api/jobs', files={'audio':(args.audio.name, handle)}, data={'title':'Real pipeline verification','backend':'ensemble','max_attempts':'1'})
        response.raise_for_status()
        assert response.status_code == 202
        job_id = response.json()['id']
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            response = client.get('/api/jobs/' + job_id)
            response.raise_for_status()
            job = response.json()
            if job['status'] in {'complete','failed'}:
                break
            time.sleep(1)
        assert job['status'] == 'complete', job.get('error') or job['status']
        result = job['result']
        assert result['metrics']['note_count'] > 0 and result['metrics']['chord_count'] > 0
        assert any('Guitar CRNN specialist proposed' in item for item in result['messages'])
        assert any('Basic Pitch reviewer proposed' in item for item in result['messages'])
        artifacts = {}
        for name, artifact in result['artifacts'].items():
            response = client.get(artifact['url'])
            response.raise_for_status()
            assert response.content
            artifacts[name] = len(response.content)
            if name == 'musicxml':
                score = ET.fromstring(response.content)
                assert score.tag == 'score-partwise' and score.findall('.//note')
                annotations = len(score.findall('.//harmony')) + len(score.findall('.//direction-type/words'))
                assert annotations == result['metrics']['chord_count'], 'A chord label was lost during score export'
                assert all(item.findtext('kind') != 'none' for item in score.findall('.//harmony')), 'A chord was incorrectly exported as no-chord'
        with httpx.Client(base_url=args.url) as stranger:
            assert stranger.get('/api/jobs').json() == []
            assert stranger.get('/api/jobs/' + job_id).status_code == 404
            assert stranger.get(result['artifacts']['source']['url']).status_code == 404
        report = {'status':'passed','real_models':True,'seconds':round(time.monotonic()-started,2),'metrics':result['metrics'],'artifact_bytes':artifacts,'score_chord_annotations':annotations,'session_isolation':'passed'}
        if args.browser:
            from playwright.sync_api import sync_playwright, expect
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                context = browser.new_context(viewport={'width':1440,'height':1000})
                context.add_cookies([{'name':cookie.name,'value':cookie.value,'url':args.url} for cookie in client.cookies.jar])
                page = context.new_page()
                page.goto(args.url)
                page.get_by_role('button', name='Recent scores', exact=True).click()
                page.locator('.history-open').filter(has_text='Real pipeline verification').first.click()
                expect(page.locator('.result-section')).to_be_visible()
                expect(page.locator('.score-preview-canvas svg').first).to_be_visible(timeout=60000)
                screenshot = Path('/tmp/projects-ready-evidence/fretwork-score.png')
                screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot), full_page=True)
                report['browser_notation_preview'] = 'passed'
                browser.close()
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
