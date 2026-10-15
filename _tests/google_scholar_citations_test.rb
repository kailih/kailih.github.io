require "jekyll"
require "minitest/autorun"
require "stringio"
require "tmpdir"
require_relative "../_plugins/google-scholar-citations"
require_relative "../_plugins/file-exists"
require_relative "../_plugins/inspirehep-citations"

class GoogleScholarCitationsTest < Minitest::Test
  KEY = "test-scholar:test-paper"
  OLD_RECORD = { "count" => 42, "fetched_at" => "2025-01-02T03:04:05Z" }.freeze
  Site = Struct.new(:source, :dest, :data) do
    def config
      { "source" => source }
    end
  end

  def setup
    @dir = Dir.mktmpdir("scholar-test")
    @site = new_site("first")
    @requests = []
  end

  def teardown
    FileUtils.remove_entry(@dir)
  end

  def new_site(name, records = { KEY => OLD_RECORD })
    source = File.join(@dir, name)
    FileUtils.mkdir_p(source)
    Site.new(source, File.join(source, "_site"), { "google_scholar_citations" => records })
  end

  def article(description = "Cited by 1,523")
    "<html><head><meta name='description' content='#{description}'></head>" \
      "<body><div id='gsc_oci_title'>Test paper</div></body></html>"
  end

  def with_fetch(site = @site, html: article, error: nil)
    fetch = lambda do |*args, &block|
      @requests << args
      raise error if error

      block.call(StringIO.new(html))
    end
    store = Jekyll::GoogleScholarCitationsStore.for_site(site)
    store.stub(:sleep, nil) do
      URI.stub(:open, fetch) { yield store }
    end
  end

  def render_tag(site = @site, field = "")
    Liquid::Template.parse("{% google_scholar_citations scholar paper #{field} %}").render!(
      { "scholar" => "test-scholar", "paper" => "test-paper" }, registers: { site: site }
    )
  end

  def test_success_updates_count_and_date_and_fetches_once_per_build
    with_fetch do |store|
      assert_equal "1520+", render_tag
      assert_includes render_tag(@site, "details"), "1523 Google Scholar citations"
      record = store.lookup("test-scholar", "test-paper")
      assert_equal 1523, record["count"]
      assert_in_delta Time.now.to_f, Time.iso8601(record["fetched_at"]).to_f, 5
      assert_equal 1, @requests.size
    end
  end

  def test_403_preserves_count_and_original_timestamp
    with_fetch(error: OpenURI::HTTPError.new("403 Forbidden", nil)) do |store|
      assert_equal "40+", render_tag
      assert_includes render_tag(@site, "details"), "last retrieved 2025-01-02 (UTC)"
      store.write
      saved = JSON.parse(File.read(File.join(@site.dest, Jekyll::GoogleScholarCitationsStore::PUBLISHED_PATH)))
      assert_equal OLD_RECORD, saved[KEY]
      assert_equal 1, @requests.size
    end
  end

  def test_timeout_keeps_verified_count
    with_fetch(error: Net::ReadTimeout.new("test timeout")) do |store|
      assert_equal OLD_RECORD, store.lookup("test-scholar", "test-paper")
    end
  end

  def test_missing_count_and_block_page_never_overwrite_with_zero
    [article("No citation metadata"), "<html><head><meta name='description' content='Cited by 0'></head></html>"].each_with_index do |html, index|
      site = new_site("invalid-page-#{index}")
      with_fetch(site, html: html) do |store|
        assert_equal OLD_RECORD, store.lookup("test-scholar", "test-paper")
      end
    end
  end

  def test_explicit_zero_is_valid
    with_fetch(html: article("Cited by 0")) do |store|
      assert_equal 0, store.lookup("test-scholar", "test-paper")["count"]
      assert_equal "0+", render_tag
    end
  end

  def test_open_graph_description_is_checked_when_normal_description_has_no_count
    html = article("No citation metadata").sub("</head>", "<meta property='og:description' content='Cited by 73'></head>")
    with_fetch(html: html) { |store| assert_equal 73, store.lookup("test-scholar", "test-paper")["count"] }
  end

  def test_without_history_actual_bibliography_template_renders_link_only
    site = new_site("no-history", {})
    with_fetch(site, error: OpenURI::HTTPError.new("403 Forbidden", nil)) do
      template = Liquid::Template.parse(File.read(File.expand_path("../_layouts/bib.liquid", __dir__)))
      html = template.render!({
        "site" => { "enable_publication_badges" => { "google_scholar" => true },
                    "data" => { "socials" => { "scholar_userid" => "test-scholar" } } },
        "entry" => { "google_scholar_id" => "test-paper", "author_array" => [] }
      }, registers: { site: site })
      link = Nokogiri::HTML.fragment(html).at_css('a[aria-label="Google Scholar link"]')
      assert_equal "Google Scholar", link.text.strip
      assert_includes link["href"], "citation_for_view=test-scholar:test-paper"
      assert_nil link.at_css("img")
      refute_includes html, "N/A"
    end
  end

  def test_published_snapshot_survives_a_fresh_build_and_failed_fetch
    with_fetch { render_tag }
    Jekyll::Hooks.trigger(:site, :post_write, @site)
    fresh_site = new_site("fresh-build")
    restored = File.join(fresh_site.source, Jekyll::GoogleScholarCitationsStore::CACHE_PATH)
    FileUtils.mkdir_p(File.dirname(restored))
    FileUtils.cp(File.join(@site.dest, Jekyll::GoogleScholarCitationsStore::PUBLISHED_PATH), restored)
    with_fetch(fresh_site, error: Net::ReadTimeout.new) do |store|
      assert_equal "1520+", render_tag(fresh_site)
      assert_equal 1523, store.lookup("test-scholar", "test-paper")["count"]
    end
  end

  def test_bad_snapshot_does_not_replace_verified_seed
    ["not JSON", JSON.generate(KEY => { "count" => "N/A", "fetched_at" => "2026-01-01T00:00:00Z" }),
     JSON.generate(KEY => { "count" => 99, "fetched_at" => "unknown" })].each_with_index do |json, index|
      site = new_site("bad-cache-#{index}")
      path = File.join(site.source, Jekyll::GoogleScholarCitationsStore::CACHE_PATH)
      FileUtils.mkdir_p(File.dirname(path))
      File.write(path, json)
      with_fetch(site, error: Net::ReadTimeout.new) { |store| assert_equal OLD_RECORD, store.lookup("test-scholar", "test-paper") }
    end
  end
end
