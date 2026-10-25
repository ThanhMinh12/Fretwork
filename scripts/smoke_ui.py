"""Browser regressions for the React UI, using controlled API fixtures, NOT ML inference.

Build web/, serve it with Vite preview, then run:
  python scripts/smoke_ui.py --url http://127.0.0.1:4175
Requires Playwright and its Chromium browser. No recordings leave the browser test.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


SCORE = '''<?xml version="1.0" encoding="utf-8"?>
<score-partwise version="4.0"><work><work-title>Practice guitar</work-title></work>
<part-list><score-part id="P1"><part-name>Guitar</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>1</divisions>
<key><fifths>0</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time>
<clef><sign>G</sign><line>2</line></clef></attributes>
<harmony><root><root-step>C</root-step></root><kind>major</kind></harmony>
<note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
<note><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
<note><pitch><step>G</step><octave>4</octave></pitch><duration>2</duration><type>half</type></note>
</measure></part></score-partwise>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:4175')
    parser.add_argument('--screenshots', type=Path)
    args = parser.parse_args()
    if args.screenshots:
        args.screenshots.mkdir(parents=True, exist_ok=True)
    state = {'jobs': [], 'polls': 0, 'runtime_requests': 0, 'reject': False, 'fail': False}
    job = {
        'id': 'ui-fixture', 'title': 'Practice guitar', 'source_filename': 'practice.wav',
        'status': 'queued', 'phase': 'Waiting for a worker', 'progress': 0,
        'error': None, 'created_at': '2026-10-18T12:00:00Z', 'updated_at': '2026-10-18T12:00:00Z',
        'options': {'backend': 'ensemble', 'capo': 'auto', 'max_span': 4, 'max_attempts': 2, 'render_pdf': False},
        'result': None,
    }
    artifacts = {name: {'filename': f'practice.{extension}', 'url': f'/api/jobs/ui-fixture/artifacts/{name}'}
                 for name, extension in [('musicxml', 'musicxml'), ('tab', 'txt'), ('source', 'wav'), ('events', 'json'), ('chords', 'json'), ('report', 'json')]}
    result = {
        'metrics': {'heuristic_health': .9, 'selected_capo': 0, 'assigned_ratio': 1, 'note_count': 3, 'chord_count': 1, 'bpm': 90, 'duration_seconds': 30},
        'profile': 'fixture', 'backend': 'ensemble', 'issues': [], 'messages': ['Controlled UI fixture; no model inference.'],
        'tab_preview': 'e|--0--3--5--|\nB|-----------|\nG|-----------|',
        'chords': [{'start': 0, 'end': 2, 'symbol': 'C', 'shape_chord': 'C'}], 'artifacts': artifacts,
    }

    def handle(route):
        path = urlparse(route.request.url).path
        method = route.request.method
        if path == '/api/runtime':
            state['runtime_requests'] += 1
            route.fulfill(json={'queued_inference': True, 'max_upload_mb': 16, 'max_audio_seconds': 30, 'temporary': True})
        elif path == '/api/jobs' and method == 'POST':
            if state['reject']:
                route.fulfill(status=503, json={'detail': 'The worker is temporarily busy. Try again.'})
                return
            body = (route.request.post_data_buffer or b'').decode('utf-8', errors='replace')
            assert 'name="backend"\r\n\r\nensemble' in body
            assert 'name="max_attempts"\r\n\r\n2' in body
            state['jobs'] = [job]
            state['polls'] = 0
            job.update(status='queued', phase='Waiting for a worker', progress=0, result=None, error=None)
            route.fulfill(status=202, json=job)
        elif path == '/api/jobs':
            route.fulfill(json=state['jobs'])
        elif path == '/api/jobs/ui-fixture' and method == 'DELETE':
            state['jobs'] = []
            route.fulfill(status=204)
        elif path == '/api/jobs/ui-fixture':
            state['polls'] += 1
            if state['polls'] == 1:
                job.update(status='processing', phase='Transcribing notes', progress=.5)
            elif state['fail']:
                job.update(status='failed', error='This recording could not be transcribed.', progress=0)
            else:
                job.update(status='complete', phase='Ready', progress=1, result=result)
            route.fulfill(json=job)
        elif '/artifacts/' in path:
            name = path.rsplit('/', 1)[-1]
            route.fulfill(body=SCORE if name == 'musicxml' else 'Controlled UI fixture',
                          content_type='application/xml' if name == 'musicxml' else 'text/plain',
                          headers={'Content-Disposition': f'attachment; filename="{artifacts[name]["filename"]}"'})
        else:
            route.fulfill(status=404, json={'detail': 'Unknown fixture route'})

    def responsive(page, name):
        for width in (1440, 1024, 768, 390, 320):
            page.set_viewport_size({'width': width, 'height': 900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1'), f'{name} overflows at {width}px'
            if name == 'empty':
                story = page.locator('.welcome-story').bounding_box()
                action = page.locator('.welcome-action').bounding_box()
                assert story['x'] < action['x'] if width > 850 else story['y'] + story['height'] < action['y']
                button = page.locator('.dropzone').bounding_box()
                assert button['y'] + button['height'] <= 900, f'Upload action is below the fold at {width}px'
            if args.screenshots and width in (1440, 390):
                page.screenshot(path=str(args.screenshots / f'fretwork-{name}-{width}.png'), full_page=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={'width': 1440, 'height': 900}, reduced_motion='reduce')
        page.set_default_timeout(15000)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.route('**/api/**', handle)
        page.goto(args.url)
        expect(page.get_by_role('heading', name='Audio to sheet music')).to_be_visible()
        expect(page.locator('.welcome-story h1')).to_have_text('Guitar transcription')
        expect(page).to_have_title('Fretwork — Guitar transcription')
        assert 'Your music.' not in page.locator('body').inner_text()
        assert page.locator('html').evaluate('(node) => getComputedStyle(node).backgroundColor') == 'rgb(250, 249, 246)'
        expect(page.locator('h1')).to_have_count(1)
        expect(page.locator('.welcome-description')).to_have_count(0)
        expect(page.locator('.brand-logo')).to_have_js_property('complete', True)
        assert page.locator('.brand-logo').evaluate('(node) => node.naturalWidth') > 0
        expect(page.locator('link[rel="icon"]')).to_have_attribute('href', '/assets/brandmark.svg')
        expect(page.get_by_text('Up to 30 seconds / 16 MB', exact=True)).to_be_visible()
        expect(page.get_by_role('button', name='Create sheet music', exact=True)).to_have_count(0)
        expect(page.locator('.advanced')).to_have_count(0)
        expect(page.locator('aside')).to_have_count(0)
        assert page.locator('main button:visible').count() == 1, 'The opening screen should have one main action'
        expect(page.get_by_label('Score title')).not_to_be_visible()
        responsive(page, 'empty')
        page.locator('.help-menu > summary').click()
        expect(page.get_by_role('link', name='Source code', exact=False)).to_be_visible()
        page.locator('.help-menu > summary').click()
        page.get_by_label('Upload guitar recording').set_input_files({'name': 'wrong.txt', 'mimeType': 'text/plain', 'buffer': b'invalid'})
        expect(page.get_by_role('alert')).to_contain_text('Choose a WAV')
        page.get_by_label('Upload guitar recording').set_input_files({'name': 'practice.wav', 'mimeType': 'audio/wav', 'buffer': b'controlled fixture'})
        expect(page.locator('.welcome-story')).to_have_count(0)
        expect(page.get_by_role('heading', level=1)).to_have_text('Audio to sheet music')
        page.locator('.advanced > summary').click()
        page.get_by_label('Score title').fill('Practice guitar')
        page.get_by_label('Review passes').select_option('2')
        responsive(page, 'advanced')
        page.locator('.advanced > summary').click()
        page.get_by_role('button', name='Create sheet music', exact=True).click()
        expect(page.get_by_role('progressbar')).to_be_visible()
        expect(page.locator('.create-section')).to_have_count(0)
        expect(page.get_by_role('button', name='New transcription', exact=True)).to_be_disabled()
        expect(page.locator('.score-preview-canvas svg').first).to_be_visible(timeout=30000)
        expect(page.get_by_text('Capo: none', exact=False)).not_to_be_visible()
        expect(page.locator('aside')).to_have_count(0)
        expect(page.locator('.create-section')).to_have_count(0)
        responsive(page, 'score')
        page.get_by_role('button', name='Guitar TAB', exact=True).click()
        expect(page.locator('.tab-preview')).to_contain_text('e|--0--3--5--|')
        page.get_by_role('button', name='Sheet music', exact=True).click()
        expect(page.locator('.score-preview-canvas svg').first).to_be_visible()
        with page.expect_download() as download:
            page.get_by_role('link', name='Download score', exact=True).click()
        assert download.value.suggested_filename == 'practice.musicxml'
        page.get_by_role('button', name='Details', exact=True).click()
        expect(page.get_by_text('Capo: none', exact=False)).to_be_visible()
        expect(page.get_by_text('This is a heuristic', exact=False)).to_be_visible()
        page.locator('.download-menu > summary').click()
        expect(page.get_by_role('link', name='Chord timeline', exact=True)).to_be_visible()
        page.get_by_role('button', name='Recent scores', exact=True).click()
        responsive(page, 'history-on-request')
        page.locator('.history-open').first.click()
        expect(page.locator('aside')).to_have_count(0)
        page.get_by_role('button', name='Sheet music', exact=True).click()
        expect(page.locator('.score-preview-canvas svg').first).to_be_visible()
        page.get_by_role('button', name='Recent scores', exact=True).click()
        expect(page.get_by_role('button', name='Delete Practice guitar')).to_be_visible()
        page.on('dialog', lambda dialog: dialog.accept())
        page.get_by_role('button', name='Delete Practice guitar').click()
        expect(page.locator('.result-section')).to_have_count(0)
        expect(page.locator('aside')).to_have_count(0)
        state['reject'] = True
        page.get_by_role('button', name='Create sheet music', exact=True).click()
        expect(page.get_by_role('alert')).to_contain_text('temporarily busy')
        state['reject'] = False
        state['fail'] = True
        page.get_by_role('button', name='Create sheet music', exact=True).click()
        expect(page.get_by_role('heading', name='Couldn’t create this score')).to_be_visible()
        page.get_by_role('button', name='Try again', exact=True).click()
        expect(page.get_by_role('button', name='Create sheet music', exact=True)).to_be_enabled()
        assert state['runtime_requests'] == 1, state
        assert not errors, errors
        browser.close()
    print(json.dumps({'status': 'passed', 'controlled_api_fixtures': True, 'real_model_inference': False, 'widths': [1440, 1024, 768, 390, 320], 'checks': ['logo and favicon', 'branded headline with one visible upload action', 'headline gives way to the selected file', 'settings only after file selection', 'upload replaced by progress and result', 'history only on request', 'file validation', 'polling', 'engraved score', 'TAB toggle', 'download', 'details', 'mobile deletion', 'server error', 'failed job retry', 'single session bootstrap']}))


if __name__ == '__main__':
    main()
